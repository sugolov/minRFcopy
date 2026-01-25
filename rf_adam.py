# implementation of Rectified Flow for simple minded people like me.
import argparse

import torch
import torchmetrics
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm

import wandb

from dit import DiT_Llama
from smooth_adam import smoother, post_adam_smoother

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images
    
@torch.no_grad()
def compute_fid(rf, dataset, args, device, channels):
    from torchmetrics.image.fid import FrechetInceptionDistance
    
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    # add real images
    print("Processing real images...")
    real_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    n_real = 0
    for x, _ in tqdm(real_loader):
        x = x.to(device)
        # unnormalize and convert to [0,1]
        x = x * 0.5 + 0.5
        x = x.clamp(0, 1)
        # FID expects RGB, repeat channels if grayscale
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)
        fid.update(x, real=True)
        n_real += x.size(0)
        if n_real >= args.num_samples:
            break
    
    # generate fake images
    print("Generating samples...")
    n_fake = 0
    while n_fake < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - n_fake)
        cond = torch.randint(0, 10, (batch_size,)).to(device)
        uncond = torch.ones_like(cond) * 10
        init_noise = torch.randn(batch_size, channels, 32, 32).to(device)
        
        images = rf.sample(init_noise, cond, uncond, sample_steps=args.sample_steps, cfg=args.cfg)
        x = images[-1]
        x = x * 0.5 + 0.5
        x = x.clamp(0, 1)
        if channels == 1:
            x = x.repeat(1, 3, 1, 1)
        fid.update(x, real=False)
        n_fake += batch_size
        print(f"Generated {n_fake}/{args.num_samples}")
    
    score = fid.compute().item()
    return score

def get_run_name(args):
    if args.smooth == 'none':
        return f"rf-baseline"
    elif args.smooth == 'ema':
        name = f"ema-rho{args.smooth_rho}"
    else:
        name = f"{args.smooth}-a{args.smooth_alpha}"
    
    if args.smooth_proj:
        name += "-proj"
    if args.smooth_post_adam:
        name += "-adam"
    if args.smooth_normalize != "none":
        name += f"-{args.smooth_normalize}"
    return name

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Rectified Flow")
    # dataset
    parser.add_argument("--cifar", action="store_true", help="Use CIFAR-10 instead of MNIST")
    parser.add_argument("--datadir", type=str, default="./data")
    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    # eval
    parser.add_argument("--eval", action="store_true", help="Run FID evaluation")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for eval")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples for FID")
    parser.add_argument("--fid_every", type=int, default=0, help="Compute FID every N epochs (0 to disable)")
    # model
    parser.add_argument("--dim", type=int, default=None, help="Model dim (default: 256 for CIFAR, 64 for MNIST)")
    parser.add_argument("--n_layers", type=int, default=None, help="Number of layers (default: 10 for CIFAR, 6 for MNIST)")
    parser.add_argument("--n_heads", type=int, default=None, help="Number of heads (default: 8 for CIFAR, 4 for MNIST)")
    # sampling
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-free guidance scale")
    # misc
    parser.add_argument("--no_ln", action="store_true", help="Disable logit-normal timestep sampling")
    parser.add_argument("--seed", type=int, default=None)
    # wandb
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None, help="Default: rf_{dataset_name}")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    # smoothing
    # smoothing
    parser.add_argument("--smooth", default="none", choices=["none", "window", "laplacian", "ema"])
    parser.add_argument("--smooth_post_adam", action="store_true", help="Apply smoothing post-Adam (on updates) instead of pre-Adam (on gradients)")
    parser.add_argument("--smooth_normalize", default="none", choices=["none", "rescale", "normalize_before"])
    parser.add_argument("--smooth_alpha", type=float, default=0.5)
    parser.add_argument("--smooth_rho", type=float, default=0.5)
    parser.add_argument("--smooth_rev", type=int, default=1)
    parser.add_argument("--smooth_proj", action="store_true")
    parser.add_argument("--smooth_fused", action="store_true")
    # output
    parser.add_argument("--output_dir", type=str, default="contents")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if args.cifar:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        channels = 3
        dim = args.dim or 256
        n_layers = args.n_layers or 10
        n_heads = args.n_heads or 8
    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        channels = 1
        dim = args.dim or 64
        n_layers = args.n_layers or 6
        n_heads = args.n_heads or 4

    model = DiT_Llama(channels, 32, dim=dim, n_layers=n_layers, n_heads=n_heads, num_classes=10).to(device)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model, ln=not args.no_ln)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataset = fdatasets(root=args.datadir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval:
        if args.ckpt:
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        fid_score = compute_fid(rf, dataset, args, device, channels)
        print(f"FID: {fid_score:.2f}")
        exit()

    if not args.no_wandb:
        wandb_project = args.wandb_project or f"rf_{dataset_name}"
        wandb_run_name = args.wandb_run_name or get_run_name(args)
        wandb.init(
            entity=args.wandb_entity,
            project=wandb_project,
            name=wandb_run_name,
            config=vars(args)
        )

    for epoch in range(args.epochs):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()

            if args.smooth != "none":
                if args.smooth_post_adam:
                    # post_adam_smoother handles optimizer.step() internally
                    post_adam_smoother(
                        model.layers, optimizer,
                        method=args.smooth,
                        alpha=args.smooth_alpha,
                        rho=args.smooth_rho,
                        reverse=bool(args.smooth_rev),
                        proj_only=args.smooth_proj,
                        normalize=args.smooth_normalize,
                    )
                else:
                    # Pre-Adam: smooth gradients, then step
                    smoother(
                        model.layers,
                        method=args.smooth,
                        alpha=args.smooth_alpha,
                        rho=args.smooth_rho,
                        reverse=bool(args.smooth_rev),
                        proj_only=args.smooth_proj,
                        normalize=args.smooth_normalize,
                    )
                    optimizer.step()
            else:
                optimizer.step()

            #optimizer.step()

            if not args.no_wandb:
                wandb.log({"loss": loss.item()})

            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        if not args.no_wandb:
            wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        rf.model.eval()
        with torch.no_grad():
            cond = torch.arange(0, 16).to(device) % 10
            uncond = torch.ones_like(cond) * 10

            init_noise = torch.randn(16, channels, 32, 32).to(device)
            images = rf.sample(init_noise, cond, uncond, sample_steps=args.sample_steps, cfg=args.cfg)
            
            gif = []
            for image in images:
                image = image * 0.5 + 0.5
                image = image.clamp(0, 1)
                x_as_image = make_grid(image.float(), nrow=4)
                img = x_as_image.permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                gif.append(Image.fromarray(img))

            gif[0].save(
                f"{args.output_dir}/sample_{epoch}.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )
            gif[-1].save(f"{args.output_dir}/sample_{epoch}_last.png")


        # checkpoint
        torch.save(model.state_dict(), f"{args.output_dir}/model_{epoch}.pt")

        # FID
        if args.fid_every > 0 and (epoch + 1) % args.fid_every == 0:
            fid_score = compute_fid(rf, dataset, args, device, channels)
            print(f"Epoch {epoch} FID: {fid_score:.2f}")
            if not args.no_wandb:
                wandb.log({"fid": fid_score})

        rf.model.train()