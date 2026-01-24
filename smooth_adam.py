
"""Gradient smoothing across residual blocks.

Methods:
    window:    g_i = (1-α)*g_i + (α/2)*(g_{i-1} + g_{i+1})
    laplacian: g_i = (1-α)*g_i - (α/2)*(g_{i-1} + g_{i+1})
    ema:       g_i = Σ_{j>=i} ρ^{j-i} * g_j  (or reversed)

Normalization options:
    'none': No normalization
    'rescale': Rescale smoothed gradient to original parameter gradient's norm
    'normalize_before': Normalize all gradients before smoothing, then rescale to original norm
"""

import torch
import torch.nn as nn
from typing import Sequence, Literal, Optional


def _get_params(block: nn.Module, proj_only: bool):
    """Yield parameters: only Linear if proj_only, else all."""
    if proj_only:
        for module in block.modules():
            if isinstance(module, nn.Linear):
                for param in module.parameters(recurse=False):
                    yield param
    else:
        for param in block.parameters():
            yield param


def window(blocks: Sequence[torch.nn.Module], alpha: float = 0.5, proj_only: bool = True, 
           normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none') -> None:
    """Neighbor averaging (low-pass filter).
    
    Args:
        blocks: Sequence of residual blocks.
        alpha: Smoothing strength.
        proj_only: If True, only smooth Linear layer params.
        normalize: Normalization method ('none', 'rescale', 'normalize_before').
    """
    L = len(blocks)
    
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    orig = [[None]*L for _ in range(n_params)]
    # Only store norms if we need to normalize before (since we'll lose the original)
    orig_norms = [[None]*L for _ in range(n_params)] if normalize == 'normalize_before' else None
    for i in range(L):
        for p, param in enumerate(params[i]):
            if param.grad is not None:
                orig[p][i] = param.grad.clone()
                if normalize == 'normalize_before':
                    orig_norms[p][i] = param.grad.norm()
                    # Normalize in-place
                    if orig_norms[p][i] > 0:
                        orig[p][i].div_(orig_norms[p][i])
    
    for i in range(L):
        for p, param in enumerate(params[i]):
            if orig[p][i] is None:
                continue
            
            left = orig[p][i-1] if i > 0 else None
            right = orig[p][i+1] if i < L-1 else None
            
            w_self = 1 - alpha/2 if (left is None or right is None) else 1 - alpha
            
            param.grad.copy_(orig[p][i] * w_self)
            if left is not None:
                param.grad.add_(left, alpha=alpha/2)
            if right is not None:
                param.grad.add_(right, alpha=alpha/2)
            
            # Rescale to original norm if requested
            if normalize == 'rescale':
                orig_norm = orig[p][i].norm()
                if orig_norm > 0:
                    current_norm = param.grad.norm()
                    if current_norm > 0:
                        param.grad.mul_(orig_norm / current_norm)
            elif normalize == 'normalize_before' and orig_norms[p][i] > 0:
                current_norm = param.grad.norm()
                if current_norm > 0:
                    param.grad.mul_(orig_norms[p][i] / current_norm)


def laplacian(blocks: Sequence[torch.nn.Module], alpha: float = 0.5, proj_only: bool = True,
              normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none') -> None:
    """Negative neighbor averaging (high-pass / sharpening).
    
    Args:
        blocks: Sequence of residual blocks.
        alpha: Smoothing strength.
        proj_only: If True, only smooth Linear layer params.
        normalize: Normalization method ('none', 'rescale', 'normalize_before').
    """
    L = len(blocks)
    
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    orig = [[None]*L for _ in range(n_params)]
    # Only store norms if we need to normalize before (since we'll lose the original)
    orig_norms = [[None]*L for _ in range(n_params)] if normalize == 'normalize_before' else None
    for i in range(L):
        for p, param in enumerate(params[i]):
            if param.grad is not None:
                orig[p][i] = param.grad.clone()
                if normalize == 'normalize_before':
                    orig_norms[p][i] = param.grad.norm()
                    # Normalize in-place
                    if orig_norms[p][i] > 0:
                        orig[p][i].div_(orig_norms[p][i])
    
    for i in range(L):
        for p, param in enumerate(params[i]):
            if orig[p][i] is None:
                continue
            
            left = orig[p][i-1] if i > 0 else None
            right = orig[p][i+1] if i < L-1 else None
            
            w_self = 1 - alpha/2 if (left is None or right is None) else 1 - alpha
            
            param.grad.copy_(orig[p][i] * w_self)
            if left is not None:
                param.grad.add_(left, alpha=-alpha/2)
            if right is not None:
                param.grad.add_(right, alpha=-alpha/2)
            
            # Rescale to original norm if requested
            if normalize == 'rescale':
                orig_norm = orig[p][i].norm()
                if orig_norm > 0:
                    current_norm = param.grad.norm()
                    if current_norm > 0:
                        param.grad.mul_(orig_norm / current_norm)
            elif normalize == 'normalize_before' and orig_norms[p][i] > 0:
                current_norm = param.grad.norm()
                if current_norm > 0:
                    param.grad.mul_(orig_norms[p][i] / current_norm)


def ema(blocks: Sequence[torch.nn.Module], rho: float = 0.5, reverse: bool = True, proj_only: bool = True,
        normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none') -> None:
    """Exponential moving average across blocks.
    
    Args:
        blocks: Sequence of residual blocks.
        rho: Decay rate.
        reverse: Whether to reverse the order.
        proj_only: If True, only smooth Linear layer params.
        normalize: Normalization method ('none', 'rescale', 'normalize_before').
    """
    L = len(blocks)
    
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    orig = [[None]*L for _ in range(n_params)]
    # Only store norms if we need to normalize before (since we'll lose the original)
    orig_norms = [[None]*L for _ in range(n_params)] if normalize == 'normalize_before' else None
    for i in range(L):
        for p, param in enumerate(params[i]):
            if param.grad is not None:
                orig[p][i] = param.grad.clone()
                if normalize == 'normalize_before':
                    orig_norms[p][i] = param.grad.norm()
                    # Normalize in-place
                    if orig_norms[p][i] > 0:
                        orig[p][i].div_(orig_norms[p][i])
    
    indices = range(L-1, -1, -1) if reverse else range(L)
    
    for p in range(n_params):
        acc = None
        for i in indices:
            if orig[p][i] is None:
                continue
            if acc is None:
                acc = orig[p][i].clone()
            else:
                acc.mul_(rho).add_(orig[p][i] * (1 - rho))
            
            # Store smoothed gradient
            params[i][p].grad.copy_(acc)
            
            # Rescale to original norm if requested
            if normalize == 'rescale':
                orig_norm = orig[p][i].norm()
                if orig_norm > 0:
                    current_norm = params[i][p].grad.norm()
                    if current_norm > 0:
                        params[i][p].grad.mul_(orig_norm / current_norm)
            elif normalize == 'normalize_before' and orig_norms[p][i] > 0:
                current_norm = params[i][p].grad.norm()
                if current_norm > 0:
                    params[i][p].grad.mul_(orig_norms[p][i] / current_norm)


def smoother(
    blocks: Sequence[torch.nn.Module],
    method: Literal['none', 'window', 'laplacian', 'ema'] = 'none',
    alpha: float = 0.5,
    rho: float = 0.5,
    reverse: bool = True,
    proj_only: bool = False,
    normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none',
) -> None:
    """Apply gradient smoothing across blocks.
    
    Args:
        blocks: Sequence of residual blocks to smooth gradients for.
        method: Smoothing method to use ('none', 'window', 'laplacian', 'ema').
        alpha: Smoothing strength for window/laplacian methods.
        rho: Decay rate for ema method.
        reverse: Whether to reverse the order for ema method.
        proj_only: If True, only smooth Linear layer params. If False, smooth all.
        normalize: Normalization method:
            - 'none': No normalization
            - 'rescale': Rescale smoothed gradient to original parameter gradient's norm
            - 'normalize_before': Normalize all gradients before smoothing, then rescale to original norm
    """
    if method == 'none':
        return
    elif method == 'window':
        window(blocks, alpha=alpha, proj_only=proj_only, normalize=normalize)
    elif method == 'laplacian':
        laplacian(blocks, alpha=alpha, proj_only=proj_only, normalize=normalize)
    elif method == 'ema':
        ema(blocks, rho=rho, reverse=reverse, proj_only=proj_only, normalize=normalize)
    else:
        raise ValueError(f"Unknown method: {method}")


def post_adam_smoother(
    blocks: Sequence[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    method: Literal['window', 'laplacian', 'ema'] = 'window',
    alpha: float = 0.5,
    reverse: bool = False,
    rho: float = 0.5,
    proj_only: bool = False,
    normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none',
    decouple_weight_decay: bool = False,
) -> None:
    """Post-Adam smoother, analogous to the in-gradient smoothers.
    
    Mechanism:
    1) Save current parameters (old)
    2) Run optimizer.step() (updates params)
    3) Compute per-parameter update delta = new - old
    4) Smooth the deltas across blocks using the same logic as gradient smoothing
    5) Restore params to old and apply smoothed deltas
    
    Args:
        blocks: Sequence of residual blocks.
        optimizer: The optimizer (should be Adam/AdamW).
        method/alpha/rho/reverse/proj_only/normalize: Same semantics as smoother(...),
            but applied to the parameter delta (update) instead of raw grads.
    """
    L = len(blocks)
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    # Build mapping from parameter -> (lr, weight_decay) for later decomposition of updates.
    # NOTE: This assumes AdamW-style *decoupled* weight decay, i.e. param *= (1 - lr*wd).
    # If the optimizer is not decoupled, the weight decay effect is mixed into the gradient
    # and cannot be separated cleanly here.
    group_hparams = {}
    for group in optimizer.param_groups:
        lr = group.get('lr', None)
        wd = group.get('weight_decay', 0.0)
        for p in group.get('params', []):
            group_hparams[id(p)] = (lr, wd)

    # Step 1: Store old parameter values
    old_params = [[None]*L for _ in range(n_params)]
    for i in range(L):
        for p_idx, param in enumerate(params[i]):
            if param.requires_grad:
                old_params[p_idx][i] = param.data.clone()

    # Step 2: Let Adam compute updates (this modifies param.data)
    optimizer.step()
    
    # Step 3: Compute updates.
    # If decouple_weight_decay=True, we decompose:
    #   total_delta = (new - old) = wd_delta + adam_delta
    # where wd_delta = -lr*wd*old (AdamW step weight decay), and only adam_delta is smoothed.
    updates = [[None]*L for _ in range(n_params)]  # the part we will smooth
    wd_updates = [[None]*L for _ in range(n_params)] if decouple_weight_decay else None
    # For both 'rescale' and 'normalize_before', we need the original (pre-smoothing) norm
    # of the non-WD (Adam) update.
    orig_norms = [[None]*L for _ in range(n_params)] if normalize in ('rescale', 'normalize_before') else None
    
    for i in range(L):
        for p_idx, param in enumerate(params[i]):
            if old_params[p_idx][i] is not None:
                total_update = param.data - old_params[p_idx][i]

                if decouple_weight_decay:
                    lr, wd = group_hparams.get(id(param), (None, 0.0))
                    # If lr is missing (shouldn't happen), fall back to no decomposition.
                    if lr is not None and wd and wd != 0.0:
                        wd_delta = old_params[p_idx][i].mul(-lr * wd)
                        wd_updates[p_idx][i] = wd_delta
                        update = total_update - wd_delta
                    else:
                        update = total_update
                else:
                    update = total_update

                updates[p_idx][i] = update.clone()
                if orig_norms is not None:
                    orig_norms[p_idx][i] = update.norm()
                if normalize == 'normalize_before' and orig_norms is not None:
                    if orig_norms[p_idx][i] > 0:
                        updates[p_idx][i].div_(orig_norms[p_idx][i])
    
    # Step 4: Smooth updates across blocks
    smoothed_updates = [[None]*L for _ in range(n_params)]
    
    if method == 'window':
        for i in range(L):
            for p_idx in range(n_params):
                if updates[p_idx][i] is None:
                    continue
                
                left = updates[p_idx][i-1] if i > 0 else None
                right = updates[p_idx][i+1] if i < L-1 else None
                
                w_self = 1 - alpha/2 if (left is None or right is None) else 1 - alpha
                smoothed = updates[p_idx][i] * w_self
                
                if left is not None:
                    smoothed.add_(left, alpha=alpha/2)
                if right is not None:
                    smoothed.add_(right, alpha=alpha/2)
                
                smoothed_updates[p_idx][i] = smoothed
                
    elif method == 'laplacian':
        for i in range(L):
            for p_idx in range(n_params):
                if updates[p_idx][i] is None:
                    continue
                
                left = updates[p_idx][i-1] if i > 0 else None
                right = updates[p_idx][i+1] if i < L-1 else None
                
                w_self = 1 - alpha/2 if (left is None or right is None) else 1 - alpha
                smoothed = updates[p_idx][i] * w_self
                
                if left is not None:
                    smoothed.add_(left, alpha=-alpha/2)
                if right is not None:
                    smoothed.add_(right, alpha=-alpha/2)
                
                smoothed_updates[p_idx][i] = smoothed
                
    elif method == 'ema':
        indices = range(L-1, -1, -1) if reverse else range(L)
        
        for p_idx in range(n_params):
            acc = None
            for i in indices:
                if updates[p_idx][i] is None:
                    continue
                if acc is None:
                    acc = updates[p_idx][i].clone()
                else:
                    acc.mul_(rho).add_(updates[p_idx][i] * (1 - rho))
                smoothed_updates[p_idx][i] = acc.clone()
    
    # Step 5: Apply normalization if needed
    for i in range(L):
        for p_idx in range(n_params):
            if smoothed_updates[p_idx][i] is None:
                continue
            
            if normalize == 'rescale':
                orig_norm = orig_norms[p_idx][i] if orig_norms is not None else None
                if orig_norm is not None and orig_norm > 0:
                    current_norm = smoothed_updates[p_idx][i].norm()
                    if current_norm > 0:
                        smoothed_updates[p_idx][i].mul_(orig_norm / current_norm)
            elif normalize == 'normalize_before':
                orig_norm = orig_norms[p_idx][i] if orig_norms is not None else None
                if orig_norm is not None and orig_norm > 0:
                    current_norm = smoothed_updates[p_idx][i].norm()
                    if current_norm > 0:
                        smoothed_updates[p_idx][i].mul_(orig_norm / current_norm)
    
    # Step 6: Apply (optional) unsmoothed wd update + smoothed updates
    for i in range(L):
        for p_idx, param in enumerate(params[i]):
            if smoothed_updates[p_idx][i] is not None:
                # Restore old value and apply smoothed update
                param.data.copy_(old_params[p_idx][i])
                if decouple_weight_decay and wd_updates is not None and wd_updates[p_idx][i] is not None:
                    param.data.add_(wd_updates[p_idx][i])
                param.data.add_(smoothed_updates[p_idx][i])