"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: /venv/main/bin/python train3.py
"""

import argparse
import json
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

fa3 = None


def init_flash_attention():
    global fa3
    if fa3 is not None:
        return
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_sizes: tuple[int, ...] = ()
    rope_base: int = 10000
    init_scale: float = 1.0
    x0_init: float = 0.1


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, base=config.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 * self.config.init_scale
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(self.config.x0_init)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, base=self.config.rope_base)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        assert len(config.window_sizes) == config.n_layer
        assert all(0 < window_size <= config.sequence_len for window_size in config.window_sizes)
        assert all(window_size & (window_size - 1) == 0 for window_size in config.window_sizes)
        return [(window_size, 0) for window_size in config.window_sizes]

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        lm_head_wd=0.0, embedding_wd=0.0, value_embedding_wd=0.0):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=lm_head_wd),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=embedding_wd),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=value_embedding_wd),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Baseline and window-size search settings
# ---------------------------------------------------------------------------

BASE_LR_MULT = 1.5
BASE_ADAM_BETAS = (0.8, 0.95)
DEVICE_BATCH_SIZE = 64
H100_BF16_PEAK_FLOPS = 989.5e12
POW2_WINDOW_SIZES = (64, 128, 256, 512, 1024, 2048)
BASE_WINDOW_SIZES = (256, 256, 256, 2048, 256, 256, 256, 2048)


@dataclass(frozen=True)
class SearchHParams:
    # Architecture
    depth: int = 8
    aspect_ratio: int = 64
    head_dim: int = 128
    window_sizes: tuple[int, ...] = BASE_WINDOW_SIZES
    rope_base: int = 10000

    # Optimization and schedules
    total_batch_size: int = 2**17
    embedding_lr: float = 0.6 * BASE_LR_MULT
    unembedding_lr: float = 0.004 * BASE_LR_MULT
    matrix_lr: float = 0.04 * BASE_LR_MULT
    scalar_lr: float = 0.5 * BASE_LR_MULT
    weight_decay: float = 0.1
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.7
    final_lr_frac: float = 0.05
    muon_momentum_warmup_steps: int = 300

    # Init and AdamW regularization
    init_scale: float = 1.0
    x0_init: float = 0.1
    lm_head_wd: float = 0.01
    embedding_wd: float = 0.001
    value_embedding_wd: float = 0.003


@dataclass(frozen=True)
class AblationRun:
    name: str
    hparams: SearchHParams
    note: str


def build_model_config(hparams, vocab_size):
    base_dim = hparams.depth * hparams.aspect_ratio
    model_dim = ((base_dim + hparams.head_dim - 1) // hparams.head_dim) * hparams.head_dim
    num_heads = model_dim // hparams.head_dim
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=hparams.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_sizes=tuple(hparams.window_sizes),
        rope_base=hparams.rope_base,
        init_scale=hparams.init_scale,
        x0_init=hparams.x0_init,
    )


def choose_device_batch_size(total_batch_size):
    max_batch = min(DEVICE_BATCH_SIZE, total_batch_size // MAX_SEQ_LEN)
    for batch_size in range(max_batch, 0, -1):
        tokens_per_fwdbwd = batch_size * MAX_SEQ_LEN
        if total_batch_size % tokens_per_fwdbwd == 0:
            return batch_size
    raise ValueError(f"Could not choose device batch size for total_batch_size={total_batch_size}")


def get_lr_multiplier(progress, hparams):
    p = min(max(progress, 0.0), 1.0)
    warmup_threshold = hparams.warmup_ratio
    cooldown_threshold = 1.0 - hparams.warmdown_ratio

    if warmup_threshold > 0 and p < warmup_threshold:
        return p / warmup_threshold
    if p < cooldown_threshold:
        return 1.0

    cooldown_denom = 1.0 - cooldown_threshold
    cooldown_p = (p - cooldown_threshold) / cooldown_denom if cooldown_denom > 0 else 1.0
    cooldown_p = min(max(cooldown_p, 0.0), 1.0)
    return 1.0 - cooldown_p * (1.0 - hparams.final_lr_frac)


def get_muon_momentum(step, hparams):
    warmup_steps = max(hparams.muon_momentum_warmup_steps, 1)
    frac = min(step / warmup_steps, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress, hparams):
    return hparams.weight_decay * (1 - progress)


def run_training_once(hparams, run_description, result_json=None):
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    init_flash_attention()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()

    print(f"Run: {run_description}")
    print(f"Search hparams: {asdict(hparams)}")

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    config = build_model_config(hparams, vocab_size)
    print(f"Model config: {asdict(config)}")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts['total']
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    device_batch_size = choose_device_batch_size(hparams.total_batch_size)
    tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
    grad_accum_steps = hparams.total_batch_size // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=hparams.unembedding_lr,
        embedding_lr=hparams.embedding_lr,
        scalar_lr=hparams.scalar_lr,
        adam_betas=BASE_ADAM_BETAS,
        matrix_lr=hparams.matrix_lr,
        weight_decay=hparams.weight_decay,
        lm_head_wd=hparams.lm_head_wd,
        embedding_wd=hparams.embedding_wd,
        value_embedding_wd=hparams.value_embedding_wd,
    )

    model = torch.compile(model, dynamic=False)

    train_loader = make_dataloader(tokenizer, device_batch_size, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)  # prefetch first batch

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Device batch size: {device_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress, hparams)
        muon_momentum = get_muon_momentum(step, hparams)
        muon_weight_decay = get_weight_decay(progress, hparams)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()

        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            sys.exit(1)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(hparams.total_batch_size / dt)
        mfu = 100 * num_flops_per_token * hparams.total_batch_size / dt / H100_BF16_PEAK_FLOPS
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    print()

    total_tokens = step * hparams.total_batch_size

    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, device_batch_size)

    t_end = time.time()
    steady_state_mfu = 100 * num_flops_per_token * hparams.total_batch_size * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    stats = {
        "description": run_description,
        "hparams": asdict(hparams),
        "model_config": asdict(config),
        "val_bpb": float(val_bpb),
        "training_seconds": float(total_training_time),
        "total_seconds": float(t_end - t_start),
        "peak_vram_mb": float(peak_vram_mb),
        "mfu_percent": float(steady_state_mfu),
        "total_tokens_M": float(total_tokens / 1e6),
        "num_steps": int(step),
        "num_params_M": float(num_params / 1e6),
        "depth": int(hparams.depth),
        "device_batch_size": int(device_batch_size),
        "grad_accum_steps": int(grad_accum_steps),
    }

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {hparams.depth}")
    print(f"device_batch:     {device_batch_size}")
    print(f"grad_accum_steps: {grad_accum_steps}")

    if result_json is not None:
        result_path = Path(result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    return stats


def is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def validate_window_sizes(window_sizes, depth=8):
    window_sizes = tuple(int(window_size) for window_size in window_sizes)
    assert len(window_sizes) == depth
    assert all(is_power_of_two(window_size) for window_size in window_sizes)
    assert all(window_size <= MAX_SEQ_LEN for window_size in window_sizes)
    return window_sizes


def add_window_design(designs, seen, name, window_sizes, note):
    if len(designs) >= 100:
        return
    window_sizes = validate_window_sizes(window_sizes)
    if window_sizes in seen:
        return
    seen.add(window_sizes)
    designs.append(AblationRun(name, replace(SearchHParams(), window_sizes=window_sizes), note))


def build_ablation_plan():
    designs = []
    seen = set()

    add_window_design(
        designs,
        seen,
        "baseline_train_py_windows",
        BASE_WINDOW_SIZES,
        "Current train.py window-size baseline",
    )

    for size in POW2_WINDOW_SIZES:
        add_window_design(
            designs,
            seen,
            f"uniform_{size}",
            (size,) * 8,
            "All layers use the same window size",
        )

    for small_size in (64, 128, 256):
        for large_size in (512, 1024, 2048):
            add_window_design(
                designs,
                seen,
                f"alternating_s{small_size}_l{large_size}",
                (small_size, large_size, small_size, large_size, small_size, large_size, small_size, large_size),
                "Alternating small and large windows",
            )
            add_window_design(
                designs,
                seen,
                f"paired_s{small_size}_l{large_size}",
                (small_size, small_size, large_size, large_size, small_size, small_size, large_size, large_size),
                "Two small-window layers followed by two larger-window layers",
            )

    ramp_designs = [
        (64, 64, 128, 128, 256, 256, 512, 512),
        (64, 128, 128, 256, 256, 512, 512, 1024),
        (128, 128, 256, 256, 512, 512, 1024, 1024),
        (128, 256, 256, 512, 512, 1024, 1024, 2048),
        (256, 256, 512, 512, 1024, 1024, 2048, 2048),
        (512, 512, 256, 256, 128, 128, 64, 64),
        (1024, 512, 512, 256, 256, 128, 128, 64),
        (2048, 1024, 1024, 512, 512, 256, 256, 128),
        (2048, 2048, 1024, 1024, 512, 512, 256, 256),
        (64, 128, 256, 512, 1024, 2048, 1024, 512),
        (128, 256, 512, 1024, 2048, 1024, 512, 256),
        (256, 512, 1024, 2048, 1024, 512, 256, 128),
    ]
    for idx, window_sizes in enumerate(ramp_designs, start=1):
        add_window_design(
            designs,
            seen,
            f"ramp_{idx:02d}",
            window_sizes,
            "Ramp or pyramid window-size schedule across depth",
        )

    for base_size in (128, 256):
        for spike_size in (512, 2048):
            if spike_size == base_size:
                continue
            for spike_pos in (0, 3, 7):
                window_sizes = [base_size] * 8
                window_sizes[spike_pos] = spike_size
                add_window_design(
                    designs,
                    seen,
                    f"single_spike_b{base_size}_p{spike_pos}_s{spike_size}",
                    window_sizes,
                    "One layer differs from an otherwise uniform window size",
                )

    for base_size in (64, 256):
        for spike_size in (512, 2048):
            for first_pos, second_pos in ((0, 4), (1, 5), (2, 6), (3, 7)):
                window_sizes = [base_size] * 8
                window_sizes[first_pos] = spike_size
                window_sizes[second_pos] = spike_size
                add_window_design(
                    designs,
                    seen,
                    f"double_spike_b{base_size}_p{first_pos}{second_pos}_s{spike_size}",
                    window_sizes,
                    "Two larger-window layers in an otherwise uniform stack",
                )

    for period in (2, 3, 4):
        for offset in range(period):
            for short_size in (64, 128, 256, 512):
                for long_size in POW2_WINDOW_SIZES:
                    if long_size <= short_size:
                        continue
                    window_sizes = tuple(
                        long_size if layer_idx % period == offset else short_size
                        for layer_idx in range(8)
                    )
                    add_window_design(
                        designs,
                        seen,
                        f"period{period}_off{offset}_s{short_size}_l{long_size}",
                        window_sizes,
                        "Periodic larger-window layers among smaller-window layers",
                    )

    if len(designs) != 100:
        raise RuntimeError(f"Expected 100 window-size designs, got {len(designs)}")
    return designs


def hparam_key(hparams):
    values = asdict(hparams)
    key_values = []
    for key in sorted(values):
        value = values[key]
        if isinstance(value, list):
            value = tuple(value)
        key_values.append((key, value))
    return tuple(key_values)


def format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(item) for item in value) + "]"
    return str(value)


def format_changes(hparams, baseline=None):
    if baseline is None:
        baseline = SearchHParams()
    values = asdict(hparams)
    base_values = asdict(baseline)
    changes = [
        f"{key}: {format_value(base_values[key])}->{format_value(values[key])}"
        for key in values
        if values[key] != base_values[key]
    ]
    return ", ".join(changes) if changes else "none"


def format_hparams(hparams):
    return ", ".join(f"{key}={format_value(value)}" for key, value in asdict(hparams).items())


def format_run_stats(stats):
    if stats.get("status") == "skipped":
        return stats["reason"]
    if stats.get("status") == "failed":
        extra = f", error={stats.get('error')}" if stats.get("error") else ""
        return f"FAILED return_code={stats.get('return_code')}{extra}"
    return (
        f"val_bpb={stats['val_bpb']:.6f}, "
        f"train_s={stats['training_seconds']:.1f}, "
        f"total_s={stats['total_seconds']:.1f}, "
        f"steps={stats['num_steps']}, "
        f"tokens_M={stats['total_tokens_M']:.1f}, "
        f"mfu={stats['mfu_percent']:.2f}%, "
        f"vram_mb={stats['peak_vram_mb']:.1f}, "
        f"device_batch={stats['device_batch_size']}, "
        f"grad_accum={stats['grad_accum_steps']}"
    )


def append_search_log(log_path, text):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def score_result(stats):
    value = stats.get("val_bpb")
    return float(value) if value is not None else float("inf")


def add_hparam_cli_args(cmd, hparams):
    for key, value in asdict(hparams).items():
        if key == "window_sizes":
            value = ",".join(str(window_size) for window_size in value)
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])


def run_child_training(hparams, description, run_idx, result_dir):
    result_path = result_dir / f"run_{run_idx:03d}.json"
    if result_path.exists():
        result_path.unlink()

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run",
        "--run-description",
        description,
        "--result-json",
        str(result_path),
    ]
    add_hparam_cli_args(cmd, hparams)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env)
    if completed.returncode != 0:
        return {
            "description": description,
            "hparams": asdict(hparams),
            "status": "failed",
            "return_code": completed.returncode,
            "val_bpb": None,
        }
    if not result_path.exists():
        return {
            "description": description,
            "hparams": asdict(hparams),
            "status": "failed",
            "return_code": 0,
            "error": "missing result json",
            "val_bpb": None,
        }
    return json.loads(result_path.read_text())


def verdict_for(stats, baseline_stats, key, baseline_key):
    if key == baseline_key:
        return "already in baseline / duplicate"
    if stats.get("status") == "failed":
        return "failed"
    if baseline_stats.get("status") == "failed":
        return "baseline failed"
    delta = stats["val_bpb"] - baseline_stats["val_bpb"]
    if delta < -1e-4:
        return f"worked ({delta:+.6f} bpb)"
    if delta > 1e-4:
        return f"regressed ({delta:+.6f} bpb)"
    return f"neutral ({delta:+.6f} bpb)"


def run_hparam_search(search_log):
    log_path = Path(search_log)
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parent / log_path
    result_dir = log_path.parent / f".{log_path.stem}_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    plan = build_ablation_plan()
    baseline = plan[0].hparams
    baseline_key = hparam_key(baseline)

    log_path.write_text(
        "Window-size design search log\n"
        f"started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "metric: lower val_bpb is better\n"
        "baseline: current train.py settings copied into train3.py\n"
        "mode: 100 exact per-layer window-size designs\n"
        f"allowed_window_sizes: {format_value(POW2_WINDOW_SIZES)}\n"
        f"python: {sys.executable}\n\n"
        f"BASELINE HPARAMS: {format_hparams(baseline)}\n\n",
        encoding="utf-8",
    )

    results = {}
    run_idx = 1
    all_results = []
    baseline_stats = None

    print(f"Writing ablation summary to {log_path}")
    for ablation in plan:
        key = hparam_key(ablation.hparams)
        append_search_log(log_path, f"## {ablation.name}")
        append_search_log(log_path, f"note: {ablation.note}")
        append_search_log(log_path, f"changes: {format_changes(ablation.hparams, baseline)}")
        append_search_log(log_path, f"hparams: {format_hparams(ablation.hparams)}")

        if key in results:
            stats = results[key]
            append_search_log(log_path, "status: REUSE duplicate/no-op settings")
        else:
            append_search_log(log_path, f"status: RUN {run_idx:03d}")
            stats = run_child_training(ablation.hparams, ablation.name, run_idx, result_dir)
            results[key] = stats
            run_idx += 1

        if ablation.name == "baseline_train_py_windows":
            baseline_stats = stats

        append_search_log(log_path, f"stats: {format_run_stats(stats)}")
        if baseline_stats is not None:
            append_search_log(log_path, f"verdict: {verdict_for(stats, baseline_stats, key, baseline_key)}")
        append_search_log(log_path, "")
        all_results.append((ablation, key, stats))

    ranked = sorted(all_results, key=lambda item: score_result(item[2]))
    append_search_log(log_path, "## Ranking")
    for rank, (ablation, key, stats) in enumerate(ranked, start=1):
        append_search_log(
            log_path,
            f"{rank:02d}. {ablation.name}: {format_run_stats(stats)}; "
            f"verdict={verdict_for(stats, baseline_stats, key, baseline_key)}",
        )

    append_search_log(log_path, "")
    append_search_log(log_path, "## Worked vs did not")
    for ablation, key, stats in all_results:
        if ablation.name == "baseline_train_py_windows":
            continue
        append_search_log(log_path, f"{ablation.name}: {verdict_for(stats, baseline_stats, key, baseline_key)}")

    best_ablation, _, best_stats = ranked[0]
    print("---")
    print(f"Best run: {best_ablation.name}")
    print(f"Best stats: {format_run_stats(best_stats)}")
    print(f"Search summary: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Autoresearch window-size design search runner")
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-description", default="manual run", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--search-log", default="window_size_search.log", help="File for compact per-run search results")

    defaults = SearchHParams()
    for key, value in asdict(defaults).items():
        flag = f"--{key.replace('_', '-')}"
        if key == "window_sizes":
            parser.add_argument(flag, type=parse_window_sizes, default=value, help=argparse.SUPPRESS)
        elif isinstance(value, int):
            parser.add_argument(flag, type=int, default=value, help=argparse.SUPPRESS)
        elif isinstance(value, float):
            parser.add_argument(flag, type=float, default=value, help=argparse.SUPPRESS)
        else:
            parser.add_argument(flag, default=value, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_window_sizes(text):
    if isinstance(text, tuple):
        return text
    parts = [part.strip() for part in text.replace("[", "").replace("]", "").split(",")]
    return validate_window_sizes(int(part) for part in parts if part)


def args_to_hparams(args):
    values = {key: getattr(args, key) for key in asdict(SearchHParams())}
    return SearchHParams(**values)


if __name__ == "__main__":
    args = parse_args()
    if args.single_run:
        run_training_once(
            args_to_hparams(args),
            run_description=args.run_description,
            result_json=args.result_json,
        )
    else:
        run_hparam_search(args.search_log)
