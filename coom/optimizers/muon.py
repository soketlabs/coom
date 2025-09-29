import torch
import triton
import triton.language as tl
import math
import time

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def newtonschulz(self, G, steps):
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G
        if G.size(-2) > G.size(-1):
            X = X.mT  
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X
    
    def muon_update(self, grad, momentum, beta=0.95, ns_steps=5):
        momentum.mul_(beta).add_(grad, alpha=1 - beta)
        update_vec = momentum  

        if update_vec.ndim == 1:
            update_mat = update_vec.unsqueeze(0).unsqueeze(0)
        elif update_vec.ndim == 2:
            update_mat = update_vec.unsqueeze(0)
        else:
            update_mat = update_vec  

        X = self.newtonschulz(update_mat, steps=ns_steps).squeeze(0)

        M, N = X.shape[-2], X.shape[-1]
        scale = math.sqrt(max(1.0, M / max(1, N)))
        X = X * scale
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum_factor = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]

                update = self.muon_update(grad, momentum_buffer, beta=momentum_factor, ns_steps=ns_steps)
                p.add_(update, alpha=-lr)
        return loss

@triton.jit
def _lerp_momentum_kernel(
    grad_ptr,
    mom_ptr,
    out_ptr,
    numel: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    m = tl.load(mom_ptr + offs, mask=mask, other=0.0)

    m = beta * m + (1 - beta) * g
    tl.store(mom_ptr + offs, m, mask=mask)

    tl.store(out_ptr + offs, m, mask=mask)

@triton.jit
def _norm_per_matrix_kernel(
    x_ptr,
    norm_ptr,
    stride_b,
    stride_m,
    stride_n,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    b = tl.program_id(axis=0)
    if b >= B:
        return
    numel = M * N
    acc = 0.0
    for start in range(0, numel, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        i = offs // N
        j = offs % N
        x = tl.load(x_ptr + b * stride_b + i * stride_m + j * stride_n, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc += tl.sum(x * x, axis=0)

    tl.store(norm_ptr + b, acc)

@triton.jit
def _scale_inv_kernel(
    x_ptr,
    inv_ptr,
    stride_b,
    stride_n,
    stride_m,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    b = tl.program_id(axis=0)
    if b >= B:
        return
    inv = tl.load(inv_ptr + b)
    numel = M * N
    for start in range(0,numel, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        i = offs // N
        j = offs % N
        x = tl.load(x_ptr + b * stride_b + i * stride_m + j * stride_n, mask=mask, other=0.0)
        x = x * inv
        tl.store(x_ptr + b * stride_b + i * stride_m + j * stride_n, x, mask=mask)

def fused_lerp(grad: torch.Tensor, momentum: torch.Tensor, beta: float):
    assert grad.is_cuda and momentum.is_cuda
    BLOCK_SIZE = 4096
    out = torch.empty_like(grad)
    grid = lambda META: ((grad.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _lerp_momentum_kernel[grid](grad, momentum, out, grad.numel(), beta, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    momentum.copy_(out)
    return out

def fused_frob_normalize(X: torch.Tensor, eps: float = 1e-7):
    B, M, N = X.shape
    BLOCK_SIZE = 4096
    norms2 = torch.empty((B,), dtype=torch.float32, device=X.device)
    _norm_per_matrix_kernel[(B,)](
        X, norms2, X.stride(0), X.stride(1), X.stride(2),
        B, M, N, 
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    inv = (norms2 + eps).rsqrt()
    _scale_inv_kernel[(B,)](
        X, inv, X.stride(0), X.stride(1), X.stride(2),
        B, M, N, 
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4
    )
    return X

class FusedMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, ns_steps=5, use_fused=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
        self.use_fused = use_fused

    def flatten_if_4d(self, t):
        return t.flatten(1) if t.ndim == 4 else t
    
    def newtonschulz(self, G, steps):
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G

        if G.size(-2) > G.size(-1):
            X = X.mT  

        if self.use_fused and X.is_cuda:
            X = fused_frob_normalize(X)
        else:
            X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X

        if G.size(-2) > G.size(-1):
            X = X.mT

        return X
    
    def muon_update(self, grad, momentum, beta=0.95, ns_steps=5):
        g = self.flatten_if_4d(grad)
        m = self.flatten_if_4d(momentum)
        if self.use_fused and g.is_cuda and m.is_cuda:
            update_vec = fused_lerp(g, m, beta)    
        else:
            m.mul_(beta).add_(g, alpha=1.0 - beta)
            update_vec = m
        if update_vec.ndim == 1:
            update_mat = update_vec.unsqueeze(0).unsqueeze(0)
        elif update_vec.ndim == 2:
            update_mat = update_vec.unsqueeze(0)
        else:
            update_mat = update_vec
        X = self.newtonschulz(update_mat, steps=ns_steps).squeeze(0)
        M, N = X.shape[-2], X.shape[-1]
        scale = math.sqrt(max(1.0, M / max(1, N)))
        X = X * scale
        return X


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum_factor = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]

                update = self.muon_update(grad, momentum_buffer, beta=momentum_factor, ns_steps=ns_steps)

                p.add_(update, alpha=-lr)

        return loss
