"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

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
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
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
        assert all(window_size > 0 for window_size in config.window_sizes)
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
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 8               # number of transformer layers
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_SIZES = [        # exact per-layer FA3 left-window sizes
    256,
    256,
    256,
    2048,
    256,
    256,
    256,
    2048,
]

# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~524K tokens per optimizer step
LR_MULT = 1.5           # baseline reference; beam search below schedules this value
BASE_EMBEDDING_LR = 0.6
BASE_UNEMBEDDING_LR = 0.004
BASE_MATRIX_LR = 0.04
BASE_SCALAR_LR = 0.5
EMBEDDING_LR = BASE_EMBEDDING_LR*LR_MULT      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = BASE_UNEMBEDDING_LR*LR_MULT  # learning rate for lm_head (Adam)
MATRIX_LR = BASE_MATRIX_LR*LR_MULT            # learning rate for matrix parameters (Muon)
SCALAR_LR = BASE_SCALAR_LR*LR_MULT            # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.1      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
LM_HEAD_WD = 0.01       # AdamW weight decay for lm_head
EMBEDDING_WD = 0.001    # AdamW weight decay for token embeddings
VALUE_EMBEDDING_WD = 0.003 # AdamW weight decay for value embeddings

DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)
H100_BF16_PEAK_FLOPS = 989.5e12

# ---------------------------------------------------------------------------
# Beam-search LR_MULT scheduler
# ---------------------------------------------------------------------------

BEAM_WIDTHS = (1, 3)
INITIAL_LR_MULTS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
CHILD_LR_FACTORS = (0.8, 1.0, 1.2)
BEAM_SEGMENT_STEPS = 50
BEAM_TOTAL_STEPS = 1350
BEAM_SEED = 42
CHECKPOINT_ROOT = Path(__file__).resolve().parent / "beam_search_checkpoints"


@dataclass
class BeamEntry:
    path: Path
    lr_mult: float
    avg_loss: float
    global_step: int
    lr_history: tuple[float, ...]
    last_losses: tuple[float, ...]


def build_model_config(depth, vocab_size):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_sizes=tuple(WINDOW_SIZES),
    )


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


def make_optimizer(model, lr_mult):
    return model.setup_optimizer(
        unembedding_lr=BASE_UNEMBEDDING_LR * lr_mult,
        embedding_lr=BASE_EMBEDDING_LR * lr_mult,
        scalar_lr=BASE_SCALAR_LR * lr_mult,
        adam_betas=ADAM_BETAS,
        matrix_lr=BASE_MATRIX_LR * lr_mult,
        weight_decay=WEIGHT_DECAY,
        lm_head_wd=LM_HEAD_WD,
        embedding_wd=EMBEDDING_WD,
        value_embedding_wd=VALUE_EMBEDDING_WD,
    )


def set_optimizer_lr_mult(optimizer, model_dim, lr_mult):
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    initial_lrs = [
        BASE_UNEMBEDDING_LR * lr_mult * dmodel_lr_scale,
        BASE_EMBEDDING_LR * lr_mult * dmodel_lr_scale,
        BASE_EMBEDDING_LR * lr_mult * dmodel_lr_scale,
        BASE_SCALAR_LR * lr_mult * 0.01,
        BASE_SCALAR_LR * lr_mult,
    ]
    for group, initial_lr in zip(optimizer.param_groups[:5], initial_lrs):
        group["initial_lr"] = initial_lr
        group["lr"] = initial_lr
    for group in optimizer.param_groups[5:]:
        group["initial_lr"] = BASE_MATRIX_LR * lr_mult
        group["lr"] = group["initial_lr"]


def set_step_schedules(optimizer, model_dim, lr_mult, step):
    set_optimizer_lr_mult(optimizer, model_dim, lr_mult)
    progress = min(step / BEAM_TOTAL_STEPS, 1.0)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(value) for value in obj)
    return obj


def save_checkpoint(path, model, optimizer, metadata):
    checkpoint = {
        "model": to_cpu(model.state_dict()),
        "optimizer": to_cpu(optimizer.state_dict()),
        "metadata": metadata,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    metadata = checkpoint.get("metadata", {})
    del checkpoint
    return metadata


def remove_checkpoint(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def clean_checkpoint_dir(checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for path in checkpoint_dir.glob("*.pt"):
        path.unlink()


def load_segment_batches(train_loader, grad_accum_steps):
    segment_batches = []
    for _ in range(BEAM_SEGMENT_STEPS):
        micro_batches = []
        for _ in range(grad_accum_steps):
            x, y, epoch = next(train_loader)
            micro_batches.append((x.detach().cpu().clone(), y.detach().cpu().clone(), epoch))
        segment_batches.append(micro_batches)
    return segment_batches


def train_candidate_segment(model, compiled_model, optimizer, segment_batches, lr_mult,
                            start_step, grad_accum_steps, autocast_ctx, num_flops_per_token):
    model.train()
    model.zero_grad(set_to_none=True)
    losses = []
    total_dt = 0.0
    last_epoch = 0
    failed = False

    for local_step, micro_batches in enumerate(segment_batches):
        global_step = start_step + local_step
        torch.cuda.synchronize()
        t0 = time.time()
        step_loss_total = 0.0

        for x_cpu, y_cpu, epoch in micro_batches:
            x = x_cpu.to("cuda", non_blocking=True)
            y = y_cpu.to("cuda", non_blocking=True)
            last_epoch = epoch
            with autocast_ctx:
                loss = compiled_model(x, y)
            step_loss_total += loss.detach().item()
            (loss / grad_accum_steps).backward()

        train_loss_f = step_loss_total / grad_accum_steps
        if math.isnan(train_loss_f) or train_loss_f > 100:
            failed = True
            model.zero_grad(set_to_none=True)
            break

        set_step_schedules(optimizer, model.config.n_embd, lr_mult, global_step)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        losses.append(train_loss_f)

        torch.cuda.synchronize()
        dt = time.time() - t0
        total_dt += dt
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt) if dt > 0 else 0
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS if dt > 0 else 0
        avg3 = sum(losses[-3:]) / min(len(losses), 3)
        print(
            f"\r    step {global_step + 1:04d}/{BEAM_TOTAL_STEPS} | "
            f"LR_MULT {lr_mult:.6g} | loss {train_loss_f:.6f} | "
            f"avg3 {avg3:.6f} | dt {dt*1000:.0f}ms | tok/sec {tok_per_sec:,} | "
            f"mfu {mfu:.1f}% | epoch {last_epoch}    ",
            end="",
            flush=True,
        )

    print()
    if failed or len(losses) < 3:
        return float("inf"), tuple(losses), total_dt
    return sum(losses[-3:]) / 3, tuple(losses), total_dt


def reserve_child_slot(child_beam, avg_loss, beam_width):
    if len(child_beam) < beam_width:
        return True

    worst = child_beam[-1]
    if avg_loss < worst.avg_loss:
        remove_checkpoint(worst.path)
        child_beam.pop()
        return True

    return False


def format_lr_history(lr_history):
    return "[" + ", ".join(f"{value:.6g}" for value in lr_history) + "]"


def candidate_lrs_for_round(beam, round_idx):
    if round_idx == 0:
        for lr_mult in INITIAL_LR_MULTS:
            yield None, lr_mult
    else:
        for parent in beam:
            for factor in CHILD_LR_FACTORS:
                yield parent, parent.lr_mult * factor


def run_beam_search(beam_width):
    assert BEAM_TOTAL_STEPS % BEAM_SEGMENT_STEPS == 0
    t_start = time.time()
    checkpoint_dir = CHECKPOINT_ROOT / f"k{beam_width}"
    clean_checkpoint_dir(checkpoint_dir)

    torch.manual_seed(BEAM_SEED)
    torch.cuda.manual_seed(BEAM_SEED)
    torch.cuda.manual_seed_all(BEAM_SEED)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    print("===")
    print(f"Beam search k={beam_width}")
    print(f"Seed: {BEAM_SEED}")
    print(f"Total steps: {BEAM_TOTAL_STEPS}")
    print(f"Segment steps: {BEAM_SEGMENT_STEPS}")
    print(f"Initial LR_MULTs: {INITIAL_LR_MULTS}")
    print(f"Child LR factors: {CHILD_LR_FACTORS}")

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    config = build_model_config(DEPTH, vocab_size)
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

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    print(f"Device batch size: {DEVICE_BATCH_SIZE}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    optimizer = make_optimizer(model, LR_MULT)
    compiled_model = torch.compile(model, dynamic=False)

    initial_path = checkpoint_dir / "initial.pt"
    save_checkpoint(initial_path, model, optimizer, {
        "beam_width": beam_width,
        "global_step": 0,
        "lr_mult": None,
        "lr_history": [],
        "avg_loss": None,
    })
    beam = [BeamEntry(
        path=initial_path,
        lr_mult=LR_MULT,
        avg_loss=float("inf"),
        global_step=0,
        lr_history=(),
        last_losses=(),
    )]

    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    num_rounds = BEAM_TOTAL_STEPS // BEAM_SEGMENT_STEPS
    total_training_time = 0.0

    for round_idx in range(num_rounds):
        start_step = round_idx * BEAM_SEGMENT_STEPS
        end_step = start_step + BEAM_SEGMENT_STEPS
        print("---")
        print(f"Round {round_idx + 1}/{num_rounds}: steps {start_step + 1}-{end_step}")
        segment_batches = load_segment_batches(train_loader, grad_accum_steps)
        child_beam = []
        parent_paths = {entry.path for entry in beam}
        candidates = list(candidate_lrs_for_round(beam, round_idx))

        for candidate_idx, (parent, lr_mult) in enumerate(candidates, start=1):
            parent_entry = beam[0] if parent is None else parent
            print(
                f"  candidate {candidate_idx}/{len(candidates)} | "
                f"parent_step={parent_entry.global_step} | LR_MULT={lr_mult:.6g}"
            )
            load_checkpoint(parent_entry.path, model, optimizer)
            set_optimizer_lr_mult(optimizer, model.config.n_embd, lr_mult)
            avg_loss, losses, train_dt = train_candidate_segment(
                model, compiled_model, optimizer, segment_batches, lr_mult,
                start_step, grad_accum_steps, autocast_ctx, num_flops_per_token,
            )
            total_training_time += train_dt

            if math.isinf(avg_loss):
                print("    candidate failed; not checkpointing")
                continue

            if not reserve_child_slot(child_beam, avg_loss, beam_width):
                print(f"    avg_loss={avg_loss:.6f} | discarded")
                continue

            child_path = checkpoint_dir / (
                f"round_{round_idx + 1:02d}_cand_{candidate_idx:03d}_"
                f"step_{end_step:04d}_lr_{lr_mult:.8g}.pt"
            )
            lr_history = parent_entry.lr_history + (lr_mult,)
            save_checkpoint(child_path, model, optimizer, {
                "beam_width": beam_width,
                "round": round_idx + 1,
                "global_step": end_step,
                "lr_mult": lr_mult,
                "lr_history": lr_history,
                "avg_loss": avg_loss,
                "last_losses": losses[-3:],
            })
            entry = BeamEntry(
                path=child_path,
                lr_mult=lr_mult,
                avg_loss=avg_loss,
                global_step=end_step,
                lr_history=lr_history,
                last_losses=losses[-3:],
            )
            child_beam.append(entry)
            child_beam.sort(key=lambda beam_entry: beam_entry.avg_loss)
            print(f"    avg_loss={avg_loss:.6f} | kept")

        for path in parent_paths:
            remove_checkpoint(path)

        if not child_beam:
            raise RuntimeError(f"All candidates failed in round {round_idx + 1}")

        beam = child_beam
        print("  survivors:")
        for rank, entry in enumerate(beam, start=1):
            print(
                f"    #{rank}: avg_loss={entry.avg_loss:.6f}, "
                f"LR_MULT={entry.lr_mult:.6g}, path={entry.path.name}"
            )
        del segment_batches
        gc.collect()

    best = beam[0]
    print("---")
    print(f"Best k={beam_width} avg_loss: {best.avg_loss:.6f}")
    print(f"Best k={beam_width} LR_MULT schedule: {format_lr_history(best.lr_history)}")

    load_checkpoint(best.path, model, optimizer)
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(compiled_model, tokenizer, DEVICE_BATCH_SIZE)

    t_end = time.time()
    total_tokens = BEAM_TOTAL_STEPS * TOTAL_BATCH_SIZE
    steady_state_mfu = (
        100 * num_flops_per_token * TOTAL_BATCH_SIZE * BEAM_TOTAL_STEPS
        / total_training_time / H100_BF16_PEAK_FLOPS
        if total_training_time > 0 else 0
    )
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"k:                {beam_width}")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"best_avg_loss:    {best.avg_loss:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {BEAM_TOTAL_STEPS}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
    print(f"checkpoint_dir:   {checkpoint_dir}")

    return {
        "k": beam_width,
        "val_bpb": float(val_bpb),
        "best_avg_loss": float(best.avg_loss),
        "lr_history": best.lr_history,
        "checkpoint": best.path,
        "survivors": beam,
    }


def main():
    results = []
    for beam_width in BEAM_WIDTHS:
        results.append(run_beam_search(beam_width))
        gc.collect()
        torch.cuda.empty_cache()

    print("===")
    print("Beam search summary")
    for result in results:
        print(
            f"k={result['k']} | val_bpb={result['val_bpb']:.6f} | "
            f"best_avg_loss={result['best_avg_loss']:.6f} | "
            f"schedule={format_lr_history(result['lr_history'])} | "
            f"checkpoint={result['checkpoint']}"
        )


if __name__ == "__main__":
    main()
