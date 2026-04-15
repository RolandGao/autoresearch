"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: /venv/main/bin/python train4.py
"""

import argparse
from collections import deque
import json
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

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
# Baseline repeat settings
# ---------------------------------------------------------------------------

BASE_LR_MULT = 1.5
BASE_ADAM_BETAS = (0.8, 0.95)
BASE_WINDOW_SIZES = (256, 256, 256, 2048, 256, 256, 256, 2048)
BASELINE_REPEATS = 10
BASELINE_SEED = 42
DEVICE_BATCH_SIZE = 64
H100_BF16_PEAK_FLOPS = 989.5e12


@dataclass(frozen=True)
class BaselineHParams:
    # Same architecture as train.py
    depth: int = 8
    aspect_ratio: int = 64
    head_dim: int = 128
    window_sizes: tuple[int, ...] = BASE_WINDOW_SIZES
    rope_base: int = 10000

    # Same optimization and schedules as train.py
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

    # Same init and AdamW regularization as train.py
    init_scale: float = 1.0
    x0_init: float = 0.1
    lm_head_wd: float = 0.01
    embedding_wd: float = 0.001
    value_embedding_wd: float = 0.003


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
    if progress < hparams.warmup_ratio:
        return progress / hparams.warmup_ratio if hparams.warmup_ratio > 0 else 1.0
    if progress < 1.0 - hparams.warmdown_ratio:
        return 1.0
    cooldown = (1.0 - progress) / hparams.warmdown_ratio
    return cooldown * 1.0 + (1 - cooldown) * hparams.final_lr_frac


def get_muon_momentum(step, hparams):
    frac = min(step / hparams.muon_momentum_warmup_steps, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress, hparams):
    return hparams.weight_decay * (1 - progress)


def run_training_once(hparams, run_description, seed, experiment_name="manual", result_json=None):
    t_start = time.time()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    init_flash_attention()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()

    print(f"Run: {run_description}")
    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed}")
    print(f"Baseline hparams: {asdict(hparams)}")

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

    smooth_train_loss = 0
    last_10_train_losses = deque(maxlen=10)
    total_training_time = 0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(grad_accum_steps):
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
        last_10_train_losses.append(train_loss_f)

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
    last_10_train_loss_mean = statistics.fmean(last_10_train_losses)
    last_10_train_loss_stdev = statistics.stdev(last_10_train_losses) if len(last_10_train_losses) > 1 else 0.0

    stats = {
        "experiment": experiment_name,
        "description": run_description,
        "seed": int(seed),
        "hparams": asdict(hparams),
        "model_config": asdict(config),
        "val_bpb": float(val_bpb),
        "training_seconds": float(total_training_time),
        "total_seconds": float(t_end - t_start),
        "peak_vram_mb": float(peak_vram_mb),
        "mfu_percent": float(steady_state_mfu),
        "last_10_train_loss_mean": float(last_10_train_loss_mean),
        "last_10_train_loss_stdev": float(last_10_train_loss_stdev),
        "last_10_train_losses": [float(value) for value in last_10_train_losses],
        "total_tokens_M": float(total_tokens / 1e6),
        "num_steps": int(step),
        "num_params_M": float(num_params / 1e6),
        "depth": int(hparams.depth),
        "device_batch_size": int(device_batch_size),
        "grad_accum_steps": int(grad_accum_steps),
    }

    print("---")
    print(f"seed:             {seed}")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"last10_loss_mean: {last_10_train_loss_mean:.6f}")
    print(f"last10_loss_std:  {last_10_train_loss_stdev:.6f}")
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


def format_run_stats(stats):
    if stats.get("status") == "failed":
        extra = f", error={stats.get('error')}" if stats.get("error") else ""
        return f"FAILED return_code={stats.get('return_code')}{extra}"
    return (
        f"seed={stats['seed']}, "
        f"val_bpb={stats['val_bpb']:.6f}, "
        f"last10_loss={stats.get('last_10_train_loss_mean', float('nan')):.6f}, "
        f"train_s={stats['training_seconds']:.1f}, "
        f"total_s={stats['total_seconds']:.1f}, "
        f"steps={stats['num_steps']}, "
        f"tokens_M={stats['total_tokens_M']:.1f}, "
        f"mfu={stats['mfu_percent']:.2f}%, "
        f"vram_mb={stats['peak_vram_mb']:.1f}"
    )


def safe_name(name):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)


def run_child_training(experiment_name, run_idx, seed, repeats, result_dir):
    result_path = result_dir / f"{safe_name(experiment_name)}_run_{run_idx:02d}_seed_{seed}.json"
    if result_path.exists():
        result_path.unlink()

    description = f"{experiment_name} repeat {run_idx:02d}/{repeats}"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run",
        "--run-description",
        description,
        "--experiment-name",
        experiment_name,
        "--seed",
        str(seed),
        "--result-json",
        str(result_path),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env)
    if completed.returncode != 0:
        return {
            "experiment": experiment_name,
            "description": description,
            "seed": seed,
            "hparams": asdict(BaselineHParams()),
            "status": "failed",
            "return_code": completed.returncode,
            "val_bpb": None,
        }
    if not result_path.exists():
        return {
            "experiment": experiment_name,
            "description": description,
            "seed": seed,
            "hparams": asdict(BaselineHParams()),
            "status": "failed",
            "return_code": 0,
            "error": "missing result json",
            "val_bpb": None,
        }
    return json.loads(result_path.read_text())


def summarize_metric(successful_stats, key):
    values = [float(stats[key]) for stats in successful_stats]
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_results(results):
    successful = [stats for stats in results if stats.get("status") != "failed"]
    failed = [stats for stats in results if stats.get("status") == "failed"]
    summary = {
        "successful_runs": len(successful),
        "failed_runs": len(failed),
    }
    if successful:
        summary["val_bpb"] = summarize_metric(successful, "val_bpb")
        if all("last_10_train_loss_mean" in stats for stats in successful):
            summary["last_10_train_loss_mean"] = summarize_metric(successful, "last_10_train_loss_mean")
        summary["num_steps"] = summarize_metric(successful, "num_steps")
        summary["training_seconds"] = summarize_metric(successful, "training_seconds")
    return summary


def write_summary(result_dir, experiments, repeats):
    all_results = []
    for experiment in experiments.values():
        all_results.extend(experiment["runs"])
    output = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.executable,
        "total_runs": len(all_results),
        "repeats_per_experiment": repeats,
        "baseline_seed": BASELINE_SEED,
        "hparams": asdict(BaselineHParams()),
        "experiments": experiments,
        "runs": all_results,
    }
    summary_path = result_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return summary_path


def print_experiment_summary(name, runs, summary):
    successful = [stats for stats in runs if stats.get("status") != "failed"]
    failed = [stats for stats in runs if stats.get("status") == "failed"]

    print("---")
    print(f"{name} summary")
    print(f"successful_runs: {len(successful)}")
    print(f"failed_runs:     {len(failed)}")
    if successful:
        val_bpb = summary["val_bpb"]
        print(f"val_bpb_mean:    {val_bpb['mean']:.6f}")
        print(f"val_bpb_stdev:   {val_bpb['stdev']:.6f}")
        print(f"val_bpb_min:     {val_bpb['min']:.6f}")
        print(f"val_bpb_max:     {val_bpb['max']:.6f}")
        if "last_10_train_loss_mean" in summary:
            last10 = summary["last_10_train_loss_mean"]
            print(f"last10_mean:     {last10['mean']:.6f}")
            print(f"last10_stdev:    {last10['stdev']:.6f}")
            print(f"last10_min:      {last10['min']:.6f}")
            print(f"last10_max:      {last10['max']:.6f}")
        print("per_run_val_bpb:")
        for stats in successful:
            line = f"  run {stats['description'].split()[-1]} seed {stats['seed']}: {stats['val_bpb']:.6f}"
            if "last_10_train_loss_mean" in stats:
                line += f" | last10_loss {stats['last_10_train_loss_mean']:.6f}"
            print(line)


def run_experiment(experiment_name, seeds, result_dir):
    repeats = len(seeds)
    results = []
    for run_idx, seed in enumerate(seeds, start=1):
        print("---")
        print(f"Starting {experiment_name} repeat {run_idx}/{repeats} with seed {seed}")
        stats = run_child_training(experiment_name, run_idx, seed, repeats, result_dir)
        print(f"Finished {experiment_name} repeat {run_idx}/{repeats}: {format_run_stats(stats)}")
        results.append(stats)
    summary = summarize_results(results)
    print_experiment_summary(experiment_name, results, summary)
    return {
        "seeds": list(seeds),
        "runs": results,
        "summary": summary,
    }


def run_baseline_experiments(repeats, result_dir, experiment_filter):
    if repeats < 1:
        raise ValueError("--repeats must be at least 1")

    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = Path(__file__).resolve().parent / result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    experiment_specs = {
        "seed_sweep": [BASELINE_SEED + i for i in range(repeats)],
        "same_seed": [BASELINE_SEED] * repeats,
    }
    if experiment_filter != "all":
        experiment_specs = {experiment_filter: experiment_specs[experiment_filter]}

    experiments = {}
    for experiment_name, seeds in experiment_specs.items():
        experiments[experiment_name] = run_experiment(experiment_name, seeds, result_dir)

    summary_path = write_summary(result_dir, experiments, repeats)

    failed = [
        stats
        for experiment in experiments.values()
        for stats in experiment["runs"]
        if stats.get("status") == "failed"
    ]

    print("---")
    print(f"total_runs:      {sum(len(experiment['runs']) for experiment in experiments.values())}")
    print(f"summary_json:    {summary_path}")

    if failed:
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline repeat experiments and report run-to-run standard deviation")
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-description", default="manual baseline run", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-name", default="manual", help=argparse.SUPPRESS)
    parser.add_argument("--result-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=BASELINE_SEED, help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=BASELINE_REPEATS, help="Number of runs per experiment")
    parser.add_argument("--experiment", choices=("all", "seed_sweep", "same_seed"), default="all", help="Which repeat experiment to run")
    parser.add_argument("--result-dir", default=".baseline_stddev_results", help="Directory for per-run JSON files and summary")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.single_run:
        run_training_once(
            BaselineHParams(),
            run_description=args.run_description,
            seed=args.seed,
            experiment_name=args.experiment_name,
            result_json=args.result_json,
        )
    else:
        run_baseline_experiments(args.repeats, args.result_dir, args.experiment)
