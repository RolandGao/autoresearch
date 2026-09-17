"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import math
import sys
import time
import uuid
from dataclasses import dataclass, asdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel

cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = (
    "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
)
fa3 = get_kernel(repo).flash_attn_interface

from prepare import EVAL_TOKENS, MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

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


def activation_l2_norm(x):
    x = x.detach().float().flatten(2)
    return x.norm(dim=-1).mean()


def log_activation_norm(activation_norms, name, x):
    if activation_norms is not None:
        activation_norms[name] = activation_l2_norm(x)


def log_residual_mix_fractions(residual_mix_l2_fractions, prefix, resid_term, x0_term):
    if residual_mix_l2_fractions is None:
        return
    resid_norm = activation_l2_norm(resid_term)
    x0_norm = activation_l2_norm(x0_term)
    total_norm = (resid_norm + x0_norm).clamp_min(1e-12)
    residual_mix_l2_fractions[f"{prefix}.resid"] = resid_norm / total_norm
    residual_mix_l2_fractions[f"{prefix}.x0"] = x0_norm / total_norm


def log_residual_path_fractions(
    residual_path_l2_fractions, prefix, branch_name, x_term, branch_term
):
    if residual_path_l2_fractions is None:
        return
    x_norm = activation_l2_norm(x_term)
    branch_norm = activation_l2_norm(branch_term)
    total_norm = (x_norm + branch_norm).clamp_min(1e-12)
    residual_path_l2_fractions[f"{prefix}.{branch_name}.x"] = x_norm / total_norm
    residual_path_l2_fractions[f"{prefix}.{branch_name}.out"] = branch_norm / total_norm


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
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin, window_size, activation_norms=None, prefix=""):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            log_activation_norm(activation_norms, f"{prefix}.attn.ve", ve)
            log_activation_norm(activation_norms, f"{prefix}.attn.v_before_ve", v)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
            log_activation_norm(activation_norms, f"{prefix}.attn.v_after_ve", v)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        log_activation_norm(activation_norms, f"{prefix}.attn.c_proj_in", y)
        y = self.c_proj(y)
        log_activation_norm(activation_norms, f"{prefix}.attn.c_proj_out", y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x, activation_norms=None, prefix=""):
        x = self.c_fc(x)
        log_activation_norm(activation_norms, f"{prefix}.mlp.c_fc_out", x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        log_activation_norm(activation_norms, f"{prefix}.mlp.c_proj_out", x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x,
        ve,
        cos_sin,
        window_size,
        activation_norms=None,
        residual_path_l2_fractions=None,
        prefix="",
    ):
        attn_out = self.attn(
            norm(x), ve, cos_sin, window_size, activation_norms, prefix
        )
        log_residual_path_fractions(
            residual_path_l2_fractions, prefix, "attn", x, attn_out
        )
        x = x + attn_out
        mlp_out = self.mlp(norm(x), activation_norms, prefix)
        log_residual_path_fractions(
            residual_path_l2_fractions, prefix, "mlp", x, mlp_out
        )
        x = x + mlp_out
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
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
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
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
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        lm_head_wd=0.0,
        embedding_wd=0.0,
        value_embedding_wd=0.0,
    ):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
        )
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.5g}")
        param_groups = [
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=lm_head_wd,
            ),
            dict(
                kind="adamw",
                params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=embedding_wd,
            ),
            dict(
                kind="adamw",
                params=value_embeds_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=value_embedding_wd,
            ),
            dict(
                kind="adamw",
                params=resid_params,
                lr=scalar_lr * 0.01,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=x0_params,
                lr=scalar_lr,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                )
            )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(
        self, idx, targets=None, reduction="mean", return_activation_norms=False
    ):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        activation_norms = {} if return_activation_norms else None
        residual_mix_l2_fractions = {} if return_activation_norms else None
        residual_path_l2_fractions = {} if return_activation_norms else None

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            prefix = f"h.{i}"
            resid_term = self.resid_lambdas[i] * x
            x0_term = self.x0_lambdas[i] * x0
            log_residual_mix_fractions(
                residual_mix_l2_fractions, prefix, resid_term, x0_term
            )
            x = resid_term + x0_term
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(
                x,
                ve,
                cos_sin,
                self.window_sizes[i],
                activation_norms,
                residual_path_l2_fractions,
                prefix,
            )
            if return_activation_norms:
                log_activation_norm(activation_norms, prefix, x)
        x = norm(x)

        softcap = 15
        log_activation_norm(activation_norms, "lm_head_in", x)
        logits = self.lm_head(x)
        log_activation_norm(activation_norms, "lm_head_out", logits)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            if return_activation_norms:
                return (
                    loss,
                    activation_norms,
                    residual_mix_l2_fractions,
                    residual_path_l2_fractions,
                )
            return loss
        if return_activation_norms:
            return (
                logits,
                activation_norms,
                residual_mix_l2_fractions,
                residual_path_l2_fractions,
            )
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
def adamw_step_fused(
    p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t
):
    decay_update = p * (lr_t * wd_t)
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    update = exp_avg / denom * step_size
    p.add_(update, alpha=-1)
    return decay_update + update


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    wd_t,
    beta2_t,
    ns_steps,
    red_dim,
):
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
    # TODO: come back
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(
        v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2
    )
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    update = lr * g
    decay_update = lr * wd * stacked_params * mask
    update = update + decay_update
    stacked_params.sub_(update)
    return update


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

    def _step_adamw(self, group, collect_update_norms=False):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            update = adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )
            if collect_update_norms:
                p.grad = update.to(dtype=p.dtype)

    def _step_muon(self, group, collect_update_norms=False):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                num_params, *shape, dtype=dtype, device=device
            )
        if "second_momentum_buffer" not in state:
            state_shape = (
                (num_params, shape[-2], 1)
                if shape[-2] >= shape[-1]
                else (num_params, 1, shape[-1])
            )
            state["second_momentum_buffer"] = torch.zeros(
                state_shape, dtype=dtype, device=device
            )
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        updates = muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))
        if collect_update_norms:
            for param, update in zip(params, updates.unbind(0)):
                param.grad = update.to(dtype=param.dtype)

    @torch.no_grad()
    def step(self, collect_update_norms=False):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group, collect_update_norms)
            elif group["kind"] == "muon":
                self._step_muon(group, collect_update_norms)


# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 8  # number of transformer layers
ASPECT_RATIO = 64  # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128  # target head dimension for attention
WINDOW_SIZES = [  # exact per-layer FA3 left-window sizes
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
MAX_STEPS = 1350  # exact number of optimizer steps to train
TOTAL_BATCH_SIZE = 2**17  # ~524K tokens per optimizer step
LR_MULT = 1.5
EMBEDDING_LR = 0.6 * LR_MULT  # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004 * LR_MULT  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04 * LR_MULT  # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5 * LR_MULT  # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.1  # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95)  # Adam beta1, beta2
WARMUP_RATIO = 0.0  # fraction of steps for LR warmup
WARMDOWN_RATIO = 0.7  # fraction of steps for LR warmdown
FINAL_LR_FRAC = 0.05  # final LR as fraction of initial
LM_HEAD_WD = 0.01  # AdamW weight decay for lm_head
EMBEDDING_WD = 0.001  # AdamW weight decay for token embeddings
VALUE_EMBEDDING_WD = 0.003  # AdamW weight decay for value embeddings

DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)
ENABLE_NORM_LOGGING = False

# Line search
BEST_LR_FACTOR = 0.8
BEST_LR_EMA_MOMENTUM = 0.9
BEST_LR_MAX_SEARCH_STEPS = 40
END_LR_MULTIPLIER = 0.1
LOG_PRE_POST_LOSSES = True
LOG_PRE_UPDATE_LOSSES = False
LOG_POST_UPDATE_LOSSES = True
LOG_BEST_LR = True

torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


# Schedules (all based on fixed-step progress)


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_training_batch_losses(step, total_steps, update, losses):
    print(
        "training_batch_losses "
        f"step={step}/{total_steps} update={update} losses={json.dumps(losses)}",
        flush=True,
    )


def format_lr_loss_map(losses_by_lr):
    return {"%.8g" % lr: loss for lr, loss in sorted(losses_by_lr.items())}


def log_best_lr(step, init_lr, best_lr, best_lr_ema, best_loss, losses_by_lr):
    print(
        "best_lr "
        f"step={step} init_lr={init_lr:.8g} best_lr={best_lr:.8g} "
        f"best_lr_ema={best_lr_ema:.8g} best_loss={best_loss:.6f} "
        f"losses={json.dumps(format_lr_loss_map(losses_by_lr))}",
        flush=True,
    )


def log_applied_lr(step, total_steps, name, lr):
    print(
        f"applied_lr step={step}/{total_steps} name={name} muon_lr={lr:.8g}",
        flush=True,
    )


def log_final_eval(train_loss, val_loss, val_bpb, time_seconds):
    print(
        f"eval epoch=final train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} "
        f"time_seconds={time_seconds:.4f}",
        flush=True,
    )


def log_run_time(run, name, wall_time_seconds, cuda_time_seconds):
    print(
        f"run_time run={run} name={name} "
        f"wall_time_seconds={wall_time_seconds:.4f} "
        f"cuda_time_seconds={cuda_time_seconds:.4f}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def finite_loss_item(loss):
    return loss.item() if torch.isfinite(loss) else float("inf")


@torch.no_grad()
def evaluate_batches_loss(model, batches):
    was_training = model.training
    model.eval()
    losses = []
    try:
        for inputs, targets in batches:
            with autocast_ctx:
                loss = model(inputs, targets, reduction="mean")
            losses.append(finite_loss_item(loss.detach()))
    finally:
        model.train(was_training)
    return losses


@torch.no_grad()
def evaluate_val_loss(model):
    was_training = model.training
    model.eval()
    val_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)
    losses = []
    try:
        for _ in range(steps):
            inputs, targets, _ = next(val_loader)
            with autocast_ctx:
                loss = model(inputs, targets, reduction="mean")
            losses.append(finite_loss_item(loss.detach()))
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Line Search
# ---------------------------------------------------------------------------


def clone_state(value):
    if torch.is_tensor(value):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {k: clone_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_state(v) for v in value)
    return value


def clone_grads(model):
    return [
        None
        if p.grad is None
        else p.grad.detach().clone(memory_format=torch.preserve_format)
        for p in model.parameters()
    ]


def restore_grads(model, grads):
    for p, grad in zip(model.parameters(), grads):
        p.grad = (
            None if grad is None else grad.clone(memory_format=torch.preserve_format)
        )


def capture_training_state(model, optimizers):
    return (
        clone_state(model.state_dict()),
        [clone_state(opt.state_dict()) for opt in optimizers],
        clone_grads(model),
        model.training,
    )


def restore_training_state(model, optimizers, state):
    model_state, optimizer_states, grads, was_training = state
    model.load_state_dict(model_state)
    for opt, opt_state in zip(optimizers, optimizer_states):
        opt.load_state_dict(clone_state(opt_state))
    restore_grads(model, grads)
    model.train(was_training)


def set_muon_lr(optimizer, lr):
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            group["lr"] = lr


def get_muon_lr(optimizer):
    return next(
        group["lr"] for group in optimizer.param_groups if group["kind"] == "muon"
    )


def evaluate_candidate_lr(
    model,
    state_model,
    optimizers,
    muon_optimizer,
    batches,
    lr,
    search_state,
    losses_by_lr,
):
    if lr in losses_by_lr:
        restore_training_state(state_model, optimizers, search_state)
        return losses_by_lr[lr]

    restore_training_state(state_model, optimizers, search_state)
    set_muon_lr(muon_optimizer, lr)
    for opt in optimizers:
        opt.step()
    losses = evaluate_batches_loss(model, batches)
    losses_by_lr[lr] = (
        sum(losses) / len(losses)
        if all(math.isfinite(loss) for loss in losses)
        else float("inf")
    )
    restore_training_state(state_model, optimizers, search_state)
    return losses_by_lr[lr]


def choose_best_lr(model, state_model, optimizers, muon_optimizer, batches, init_lr):
    search_state = capture_training_state(state_model, optimizers)
    losses_by_lr = {}

    def candidate_loss(lr):
        return evaluate_candidate_lr(
            model,
            state_model,
            optimizers,
            muon_optimizer,
            batches,
            lr,
            search_state,
            losses_by_lr,
        )

    center_lr = init_lr
    for _ in range(BEST_LR_MAX_SEARCH_STEPS):
        left_lr = center_lr * BEST_LR_FACTOR
        right_lr = center_lr / BEST_LR_FACTOR
        left_loss = candidate_loss(left_lr)
        center_loss = candidate_loss(center_lr)
        right_loss = candidate_loss(right_lr)
        if center_loss <= left_loss and center_loss <= right_loss:
            restore_training_state(state_model, optimizers, search_state)
            return center_lr, center_loss, losses_by_lr
        center_lr = left_lr if left_loss < right_loss else right_lr

    best_lr, best_loss = min(losses_by_lr.items(), key=lambda item: item[1])
    restore_training_state(state_model, optimizers, search_state)
    return best_lr, best_loss, losses_by_lr


def resolve_best_lr_scheduler(best_lr_scheduler, best_lr_linear_decay):
    if best_lr_scheduler is not None:
        return best_lr_scheduler
    return "linear" if best_lr_linear_decay else "constant"


def uses_best_lr_decay(best_lr_scheduler):
    return best_lr_scheduler != "constant"


def best_lr_scheduler_multiplier(step, total_steps, steps_per_epoch, best_lr_scheduler):
    if best_lr_scheduler == "constant":
        return 1.0
    if best_lr_scheduler == "linear":
        denominator = max(1, total_steps - 1)
        progress = step / denominator
        return 1.0 + (END_LR_MULTIPLIER - 1.0) * progress
    if best_lr_scheduler == "last2_linear":
        decay_start_step = max(0, total_steps - 2 * steps_per_epoch)
        if step < decay_start_step:
            return 1.0
        denominator = max(1, total_steps - 1 - decay_start_step)
        progress = (step - decay_start_step) / denominator
        return 1.0 + (END_LR_MULTIPLIER - 1.0) * progress
    raise ValueError(f"Unknown best_lr_scheduler: {best_lr_scheduler}")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def set_training_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_sizes=tuple(WINDOW_SIZES),
    )


def setup_optimizer(model):
    return model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
        lm_head_wd=LM_HEAD_WD,
        embedding_wd=EMBEDDING_WD,
        value_embedding_wd=VALUE_EMBEDDING_WD,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    base_model = GPT(config)
base_model.to_empty(device=device)
base_model.init_weights()

param_counts = base_model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = base_model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:.4e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
print(f"Training steps: {MAX_STEPS}")
print(f"Gradient accumulation steps: {grad_accum_steps}")


def fixed_muon_run_config():
    return dict(
        name=f"muon_lr{MATRIX_LR:.6g}",
        muon_lr=MATRIX_LR,
        best_lr_strategy=None,
    )


def best_lr_run_config(schedule_name, scheduler):
    return dict(
        name=f"best_lr_{schedule_name}_lr{MATRIX_LR:.6g}",
        muon_lr=MATRIX_LR,
        best_lr_strategy="min_loss",
        best_lr_scheduler=scheduler,
    )


RUN_CONFIGS = [
    fixed_muon_run_config(),
    best_lr_run_config("constant", "constant"),
    best_lr_run_config("decay1to0.1", "linear"),
]


def main(
    run,
    model,
    name,
    muon_lr,
    best_lr_strategy=None,
    best_lr_linear_decay=False,
    best_lr_scheduler=None,
):
    run_id = run
    run_wall_start = time.perf_counter()
    state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    set_training_seed()
    use_best_lr = best_lr_strategy is not None
    best_lr_scheduler = resolve_best_lr_scheduler(
        best_lr_scheduler, best_lr_linear_decay
    )
    best_lr_linear_decay = uses_best_lr_decay(best_lr_scheduler)

    state_model.init_weights()
    optimizer = setup_optimizer(state_model)
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            group["lr"] = muon_lr
            group["initial_lr"] = muon_lr
    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0

    def start_timer():
        starter.record()

    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0
    best_lr_ema = muon_lr
    best_lr_logs = []
    training_batch_loss_logs = []
    loss_log_steps_list = [MAX_STEPS] if LOG_PRE_POST_LOSSES else []
    loss_log_steps = set(loss_log_steps_list)
    final_post_update_losses = None

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        start_timer()
        line_search_batches = []
        for _ in range(grad_accum_steps):
            line_search_batches.append((x.detach().clone(), y.detach().clone()))
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        progress = min(step / max(1, MAX_STEPS - 1), 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        if not use_best_lr:
            set_muon_lr(optimizer, muon_lr * lrm)

        if LOG_PRE_UPDATE_LOSSES and step + 1 in loss_log_steps:
            stop_timer()
            pre_update_losses = evaluate_batches_loss(model, line_search_batches)
            training_batch_loss_logs.append(
                dict(step=step + 1, update="pre", losses=pre_update_losses)
            )
            log_training_batch_losses(
                step + 1, MAX_STEPS, "pre", pre_update_losses
            )
            start_timer()

        if use_best_lr:
            lr_decay_multiplier = best_lr_scheduler_multiplier(
                step, MAX_STEPS, MAX_STEPS, best_lr_scheduler
            )
            init_lr = best_lr_ema
            if best_lr_strategy != "min_loss":
                raise ValueError(f"Unknown best_lr_strategy: {best_lr_strategy}")
            searched_lr, best_loss, losses_by_lr = choose_best_lr(
                model,
                state_model,
                [optimizer],
                optimizer,
                line_search_batches,
                init_lr,
            )
            actual_lr = searched_lr * lr_decay_multiplier
            best_lr_ema = (
                BEST_LR_EMA_MOMENTUM * best_lr_ema
                + (1 - BEST_LR_EMA_MOMENTUM) * searched_lr
            )
            best_lr_logs.append(
                dict(
                    step=step + 1,
                    strategy=best_lr_strategy,
                    init_lr=init_lr,
                    searched_lr=searched_lr,
                    actual_lr=actual_lr,
                    best_lr=searched_lr,
                    best_lr_ema=best_lr_ema,
                    best_lr_scheduler=best_lr_scheduler,
                    lr_decay_multiplier=lr_decay_multiplier,
                    best_loss=best_loss,
                    losses_by_lr=format_lr_loss_map(losses_by_lr),
                )
            )
            if LOG_BEST_LR:
                stop_timer()
                log_best_lr(
                    step + 1,
                    init_lr,
                    searched_lr,
                    best_lr_ema,
                    best_loss,
                    losses_by_lr,
                )
                start_timer()
            set_muon_lr(optimizer, actual_lr)

        applied_muon_lr = get_muon_lr(optimizer)
        log_applied_lr(step + 1, MAX_STEPS, name, applied_muon_lr)
        optimizer.step()
        state_model.zero_grad(set_to_none=True)
        step += 1

        if LOG_POST_UPDATE_LOSSES and step in loss_log_steps:
            stop_timer()
            post_update_losses = evaluate_batches_loss(model, line_search_batches)
            final_post_update_losses = post_update_losses
            training_batch_loss_logs.append(
                dict(step=step, update="post", losses=post_update_losses)
            )
            log_training_batch_losses(step, MAX_STEPS, "post", post_update_losses)
            start_timer()

        stop_timer()
        train_loss_f = train_loss.item()
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            exit(1)

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = (
            ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        )
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**step)
        pct_done = 100 * min(step / MAX_STEPS, 1.0)
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
        remaining_steps = max(0, MAX_STEPS - step)
        print(
            f"\rstep {step - 1:05d} ({pct_done:.5g}%) | "
            f"loss: {debiased_smooth_loss:.5g} | lrm: {lrm:.5g} | "
            f"dt: {dt * 1000:.5g}ms | tok/sec: {tok_per_sec:,} | "
            f"mfu: {mfu:.5g}% | epoch: {epoch} | "
            f"remaining_steps: {remaining_steps}    ",
            end="",
            flush=True,
        )

        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

        if step >= MAX_STEPS:
            break

    print()
    total_tokens = step * TOTAL_BATCH_SIZE
    train_loss = (
        sum(final_post_update_losses) / len(final_post_update_losses)
        if final_post_update_losses
        else train_loss_f
    )

    start_timer()
    val_loss = evaluate_val_loss(model)
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
    stop_timer()

    t_end = time.time()
    steady_state_mfu = (
        100
        * num_flops_per_token
        * TOTAL_BATCH_SIZE
        * (step - 10)
        / total_training_time
        / H100_BF16_PEAK_FLOPS
        if total_training_time > 0
        else 0
    )
    wall_time_seconds = time.perf_counter() - run_wall_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    log_final_eval(train_loss, val_loss, val_bpb, time_seconds)
    log_run_time(run_id, name, wall_time_seconds, time_seconds)

    print("---")
    print(f"val_bpb:          {val_bpb:.5g}")
    print(f"train_loss:       {train_loss:.5g}")
    print(f"val_loss:         {val_loss:.5g}")
    print(f"training_seconds: {total_training_time:.5g}")
    print(f"total_seconds:    {t_end - t_start_training:.5g}")
    print(f"peak_vram_mb:     {peak_vram_mb:.5g}")
    print(f"mfu_percent:      {steady_state_mfu:.5g}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.5g}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.5g}")
    print(f"depth:            {DEPTH}")

    return dict(
        train_loss=train_loss,
        val_loss=val_loss,
        val_bpb=val_bpb,
        name=name,
        muon_lr=muon_lr,
        use_best_lr=use_best_lr,
        best_lr_strategy=best_lr_strategy,
        best_lr_linear_decay=best_lr_linear_decay,
        best_lr_scheduler=best_lr_scheduler,
        best_lr_ema=best_lr_ema,
        best_lr_logs=best_lr_logs,
        training_batch_loss_logs=training_batch_loss_logs,
        training_batch_loss_log_steps=loss_log_steps_list,
        wall_time_seconds=wall_time_seconds,
        cuda_time_seconds=time_seconds,
        peak_vram_mb=peak_vram_mb,
        mfu_percent=steady_state_mfu,
        total_tokens=total_tokens,
        num_steps=step,
        depth=DEPTH,
    )


if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()

    set_training_seed()
    compiled_model = torch.compile(base_model, dynamic=False)

    results = []
    for run, run_config in enumerate(RUN_CONFIGS):
        print(
            "llm_baseline2 run=%d muon_lr=%.6g name=%s best_lr_strategy=%s "
            "best_lr_linear_decay=%s best_lr_scheduler=%s"
            % (
                run,
                run_config["muon_lr"],
                run_config["name"],
                run_config["best_lr_strategy"],
                run_config.get("best_lr_linear_decay", False),
                resolve_best_lr_scheduler(
                    run_config.get("best_lr_scheduler"),
                    run_config.get("best_lr_linear_decay", False),
                ),
            ),
            flush=True,
        )
        result = main(run, compiled_model, **run_config)
        results.append(result)
        print("Name:               %s" % result["name"])
        print("Muon lr:            %.6g" % result["muon_lr"])
        print("Use best lr:        %s" % result["use_best_lr"])
        if result["use_best_lr"]:
            print("Best lr strategy:   %s" % result["best_lr_strategy"])
            print("Best lr decay:      %s" % result["best_lr_linear_decay"])
            print("Best lr scheduler:  %s" % result["best_lr_scheduler"])
            print("Final best lr ema:  %.6g" % result["best_lr_ema"])
        print("Train loss:         %.4f" % result["train_loss"])
        print("Val loss:           %.4f" % result["val_loss"])
        print("Val bpb:            %.4f" % result["val_bpb"])

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, results=results), log_path)
    print(os.path.abspath(log_path))
