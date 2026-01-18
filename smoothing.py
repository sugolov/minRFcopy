"""Gradient smoothing across residual blocks (memory-efficient)."""
import torch
import torch.nn as nn
from typing import Sequence, Literal


def _get_params(block: nn.Module, proj_only: bool):
    if proj_only:
        for module in block.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                for param in module.parameters(recurse=False):
                    yield param
    else:
        for param in block.parameters():
            yield param

def window_foreach(blocks, alpha=0.5, proj_only=True):
    L = len(blocks)
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    prev = None  # list of n_params tensors for block i-1
    
    for i in range(L):
        curr_grads = [params[i][p].grad for p in range(n_params)]
        
        if any(g is None for g in curr_grads):
            prev = None
            continue
        
        # clone current before overwriting
        curr_orig = [g.clone() for g in curr_grads]
        
        # right is unmodified, read directly
        right = [params[i+1][p].grad for p in range(n_params)] if i < L-1 else None
        
        has_left = prev is not None
        has_right = right is not None
        w_self = 1 - alpha if (has_left and has_right) else 1 - alpha/2
        
        # apply smoothing with fused ops
        torch._foreach_mul_(curr_grads, w_self)
        if has_left:
            torch._foreach_add_(curr_grads, prev, alpha=alpha/2)
        if has_right:
            torch._foreach_add_(curr_grads, right, alpha=alpha/2)
        
        prev = curr_orig


def laplacian_foreach(blocks, alpha=0.5, proj_only=True):
    L = len(blocks)
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    prev = None
    
    for i in range(L):
        curr_grads = [params[i][p].grad for p in range(n_params)]
        
        if any(g is None for g in curr_grads):
            prev = None
            continue
        
        curr_orig = [g.clone() for g in curr_grads]
        right = [params[i+1][p].grad for p in range(n_params)] if i < L-1 else None
        
        has_left = prev is not None
        has_right = right is not None
        w_self = 1 - alpha if (has_left and has_right) else 1 - alpha/2
        
        torch._foreach_mul_(curr_grads, w_self)
        if has_left:
            torch._foreach_add_(curr_grads, prev, alpha=-alpha/2)
        if has_right:
            torch._foreach_add_(curr_grads, right, alpha=-alpha/2)
        
        prev = curr_orig


def ema_foreach(blocks, rho=0.5, reverse=True, proj_only=True):
    L = len(blocks)
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    indices = range(L-1, -1, -1) if reverse else range(L)
    acc = None
    
    for i in indices:
        curr_grads = [params[i][p].grad for p in range(n_params)]
        
        if any(g is None for g in curr_grads):
            continue
        
        if acc is None:
            acc = [g.clone() for g in curr_grads]
        else:
            torch._foreach_mul_(acc, rho)
            torch._foreach_add_(acc, curr_grads)
        
        # copy acc back to grads
        for g, a in zip(curr_grads, acc):
            g.copy_(a)


def window(blocks: Sequence[nn.Module], alpha: float = 0.5, proj_only: bool = True) -> None:
    L = len(blocks)
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    for p in range(n_params):
        prev = None  # original grad of i-1
        
        for i in range(L):
            param = params[i][p]
            if param.grad is None:
                prev = None
                continue
            
            # save current before overwriting
            curr = param.grad.clone()
            
            # right is unmodified, read directly
            right_param = params[i+1][p] if i < L-1 else None
            right = right_param.grad if right_param is not None and right_param.grad is not None else None
            
            w_self = 1 - alpha if (prev is not None and right is not None) else 1 - alpha/2
            
            param.grad.copy_(curr)
            param.grad.mul_(w_self)
            if prev is not None:
                param.grad.add_(prev, alpha=alpha/2)
            if right is not None:
                param.grad.add_(right, alpha=alpha/2)
            
            prev = curr


def laplacian(blocks: Sequence[nn.Module], alpha: float = 0.5, proj_only: bool = True) -> None:
    L = len(blocks)
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    for p in range(n_params):
        prev = None
        
        for i in range(L):
            param = params[i][p]
            if param.grad is None:
                prev = None
                continue
            
            curr = param.grad.clone()
            
            right_param = params[i+1][p] if i < L-1 else None
            right = right_param.grad if right_param is not None and right_param.grad is not None else None
            
            w_self = 1 - alpha if (prev is not None and right is not None) else 1 - alpha/2
            
            param.grad.copy_(curr)
            param.grad.mul_(w_self)
            if prev is not None:
                param.grad.add_(prev, alpha=-alpha/2)
            if right is not None:
                param.grad.add_(right, alpha=-alpha/2)
            
            prev = curr


def ema(blocks: Sequence[nn.Module], rho: float = 0.5, reverse: bool = True, proj_only: bool = True) -> None:
    """EMA only needs O(1) buffer — accumulator."""
    L = len(blocks)
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    indices = range(L-1, -1, -1) if reverse else range(L)
    
    for p in range(n_params):
        acc = None
        for i in indices:
            param = params[i][p]
            if param.grad is None:
                continue
            if acc is None:
                acc = param.grad.clone()
            else:
                acc.mul_(rho).add_(param.grad, alpha=1-rho)
            param.grad.copy_(acc)


def smooth(
    blocks: Sequence[nn.Module],
    method: Literal['none', 'window', 'laplacian', 'ema'] = 'none',
    alpha: float = 0.5,
    rho: float = 0.5,
    reverse: bool = True,
    proj_only: bool = True,
    fused: bool = False,
) -> None:
    if method == 'none':
        return
    elif method == 'window':
        if fused:
            window_foreach(blocks, alpha=alpha, proj_only=proj_only)
        else:
            window(blocks, alpha=alpha, proj_only=proj_only)
    elif method == 'laplacian':
        if fused:
            laplacian_foreach(blocks, alpha=alpha, proj_only=proj_only)
        else:
            laplacian(blocks, alpha=alpha, proj_only=proj_only)
    elif method == 'ema':
        if fused:
            ema_foreach(blocks, rho=rho, reverse=reverse, proj_only=proj_only)
        else:
            ema(blocks, rho=rho, reverse=reverse, proj_only=proj_only)
    else:
        raise ValueError(f"Unknown method: {method}")