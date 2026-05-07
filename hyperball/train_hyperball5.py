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
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

fa3 = None


def initialize_flash_attention():
    if not torch.cuda.is_available():
        raise RuntimeError("train_hyperball5.py requires CUDA.")

    capability = torch.cuda.get_device_capability()
    # varunneal's FA3 is Hopper only; use kernels-community on non-Hopper GPUs.
    repo = (
        "varunneal/flash-attention-3"
        if capability == (9, 0)
        else "kernels-community/flash-attn3"
    )
    return get_kernel(repo).flash_attn_interface


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


class ScaledLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.log_scale = nn.Parameter(torch.empty(()))

    def forward(self, input):
        out = F.linear(input, self.weight, self.bias)
        return out * self.log_scale.exp().to(dtype=out.dtype)


class ScaledEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
        self.log_scale = nn.Parameter(torch.empty(()))

    def forward(self, input):
        out = F.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return out * self.log_scale.exp().to(dtype=out.dtype)


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


def log_l2_fraction_pair(fractions, prefix, left_name, left_term, right_name, right_term):
    if fractions is None:
        return
    left_norm = activation_l2_norm(left_term)
    right_norm = activation_l2_norm(right_term)
    total_norm = (left_norm + right_norm).clamp_min(1e-12)
    fractions[f"{prefix}.{left_name}"] = left_norm / total_norm
    fractions[f"{prefix}.{right_name}"] = right_norm / total_norm


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
        self.c_q = ScaledLinear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = ScaledLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = ScaledLinear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = ScaledLinear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            ScaledLinear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

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
        if fa3 is None:
            raise RuntimeError("Flash attention must be initialized before training.")

        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            gated_ve = gate.unsqueeze(-1) * ve
            log_l2_fraction_pair(
                residual_path_l2_fractions,
                f"{prefix}.attn.ve_mix",
                "v",
                v,
                "gated_ve",
                gated_ve,
            )
            v = v + gated_ve

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
        self.c_fc = ScaledLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = ScaledLinear(4 * config.n_embd, config.n_embd, bias=False)

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
            norm(x),
            ve,
            cos_sin,
            window_size,
            activation_norms,
            residual_path_l2_fractions,
            prefix,
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
                "wte": ScaledEmbedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = ScaledLinear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): ScaledEmbedding(config.vocab_size, kv_dim)
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
        target_norm_by_kind = {
            "attn.c_proj": 5.0,
            "attn.ve_gate": 1.0,
            "k": 100,
            "lm_head": 264.192,
            "mlp.c_fc": 36.95041722813605,
            "mlp.c_proj": 5.0,
            "q": 100,
            "v": 111.9491455974542,
            "ve": 24000.0,
            "wte": 4096.0,
            "resid_lambdas": 1.0,
            "x0_lambdas": 5.0,
        }

        def init_scaled(module, target_frobenius_norm):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
            norm = module.weight.detach().float().norm().clamp_min(1e-12)
            module.weight.div_(norm.to(dtype=module.weight.dtype))
            module.log_scale.fill_(math.log(target_frobenius_norm))

        matrix_specs = list(self.scaled_matrix_module_specs())
        missing_kinds = {
            weight_kind
            for _, weight_kind, _ in matrix_specs
            if weight_kind not in target_norm_by_kind
        }
        if missing_kinds:
            raise KeyError(f"Missing target norms for weight kinds: {missing_kinds}")

        for _, weight_kind, module in matrix_specs:
            init_scaled(module, target_norm_by_kind[weight_kind])

        # Per-layer scalars
        self.resid_lambdas.fill_(target_norm_by_kind["resid_lambdas"])
        self.x0_lambdas.fill_(target_norm_by_kind["x0_lambdas"])
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.weight.data = self.transformer.wte.weight.to(
            dtype=torch.bfloat16
        )
        for ve in self.value_embeds.values():
            ve.weight.data = ve.weight.to(dtype=torch.bfloat16)

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

    def scaled_matrix_module_specs(self):
        yield "wte", "wte", self.transformer.wte
        yield "lm_head", "lm_head", self.lm_head
        for layer_idx, ve in self.value_embeds.items():
            yield f"h.{layer_idx}.ve", "ve", ve
        for layer_idx, block in enumerate(self.transformer.h):
            prefix = f"h.{layer_idx}"
            yield f"{prefix}.q", "q", block.attn.c_q
            yield f"{prefix}.k", "k", block.attn.c_k
            yield f"{prefix}.v", "v", block.attn.c_v
            yield f"{prefix}.attn.c_proj", "attn.c_proj", block.attn.c_proj
            yield f"{prefix}.mlp.c_fc", "mlp.c_fc", block.mlp.c_fc
            yield f"{prefix}.mlp.c_proj", "mlp.c_proj", block.mlp.c_proj
            if block.attn.ve_gate is not None:
                yield f"{prefix}.attn.ve_gate", "attn.ve_gate", block.attn.ve_gate

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

    def _angular_muon_weight_specs(self):
        for block in self.transformer.h:
            yield "q", block.attn.c_q.weight
            yield "k", block.attn.c_k.weight
            yield "v", block.attn.c_v.weight
            yield "attn.c_proj", block.attn.c_proj.weight
            yield "mlp.c_fc", block.mlp.c_fc.weight
            yield "mlp.c_proj", block.mlp.c_proj.weight
            if block.attn.ve_gate is not None:
                yield "attn.ve_gate", block.attn.ve_gate.weight

    def _angular_adamw_weight_specs(self):
        yield "wte", self.transformer.wte.weight
        yield "lm_head", self.lm_head.weight
        for ve in self.value_embeds.values():
            yield "ve", ve.weight

    def _log_scale_param_specs(self):
        for _, weight_kind, module in self.scaled_matrix_module_specs():
            yield weight_kind, module.log_scale

    def _scalar_param_groups(self, scalar_lr, adam_betas):
        return [
            dict(
                kind="adamw",
                params=[self.resid_lambdas],
                lr=scalar_lr * 0.01,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=[self.x0_lambdas],
                lr=scalar_lr,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]

    def _log_scale_param_groups(self, log_scale_specs):
        param_groups = []
        for weight_kind in sorted({kind for kind, _ in log_scale_specs}):
            group_params = [
                param
                for param_kind, param in log_scale_specs
                if param_kind == weight_kind
            ]
            param_groups.append(
                dict(
                    kind="adamw",
                    weight_kind=f"{weight_kind}.scale",
                    params=group_params,
                    lr=SCALE_LR_BY_WEIGHT_KIND[weight_kind],
                    betas=(0.0, 0.95),
                    eps=1e-10,
                    weight_decay=0.0,
                    scale_lr_schedule=True,
                )
            )
        return param_groups

    def _angular_muon_param_groups(self, muon_specs, matrix_lrs):
        param_groups = []
        group_keys = sorted(
            {(weight_kind, tuple(param.shape)) for weight_kind, param in muon_specs}
        )
        for weight_kind, shape in group_keys:
            group_params = [
                param
                for param_kind, param in muon_specs
                if param_kind == weight_kind and tuple(param.shape) == shape
            ]
            param_groups.append(
                dict(
                    kind="angular_muon",
                    weight_kind=weight_kind,
                    params=group_params,
                    lr=matrix_lrs[weight_kind],
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                )
            )
        return param_groups

    def _angular_adamw_param_groups(self, adamw_direction_specs, matrix_lrs, adam_betas):
        return [
            dict(
                kind="angular_adamw",
                weight_kind=weight_kind,
                params=[param],
                lr=matrix_lrs[weight_kind],
                betas=adam_betas,
                eps=1e-10,
            )
            for weight_kind, param in adamw_direction_specs
        ]

    def _assert_optimizer_covers_model_params(self, param_groups):
        grouped_params = [
            param
            for group in param_groups
            for param in group["params"]
        ]
        model_params = list(self.parameters())
        if len(grouped_params) != len(model_params):
            raise ValueError(
                "Optimizer parameter groups do not cover the model exactly: "
                f"{len(grouped_params)} grouped params vs {len(model_params)} model params."
            )
        if {id(param) for param in grouped_params} != {id(param) for param in model_params}:
            raise ValueError("Optimizer parameter groups do not match model parameters.")

    def _add_initial_lrs(self, optimizer):
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]

    def setup_optimizer(
        self,
        matrix_lrs=None,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
    ):
        if matrix_lrs is None:
            matrix_lrs = INITIAL_EFFECTIVE_LR_BY_WEIGHT_KIND

        muon_specs = list(self._angular_muon_weight_specs())
        adamw_direction_specs = list(self._angular_adamw_weight_specs())
        log_scale_specs = list(self._log_scale_param_specs())

        print(
            "Using fixed-norm angular directions for every 2D matrix "
            "with projection_init_scale=5"
        )

        param_groups = []
        param_groups.extend(self._scalar_param_groups(scalar_lr, adam_betas))
        param_groups.extend(self._log_scale_param_groups(log_scale_specs))
        param_groups.extend(self._angular_muon_param_groups(muon_specs, matrix_lrs))
        param_groups.extend(
            self._angular_adamw_param_groups(
                adamw_direction_specs, matrix_lrs, adam_betas
            )
        )

        self._assert_optimizer_covers_model_params(param_groups)
        optimizer = MuonAdamW(param_groups)
        self._add_initial_lrs(optimizer)
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

        softcap = 8.0
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

POLAR_EXPRESS_COEFFS = [
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


def angular_fallback_tangent(p):
    fallback = torch.roll(p, shifts=1, dims=-1)
    p_norm_sq = (p * p).sum(dim=(-2, -1), keepdim=True).clamp_min(1e-24)
    tangent = fallback - (fallback * p).sum(dim=(-2, -1), keepdim=True) / p_norm_sq * p
    second = torch.roll(p, shifts=1, dims=-2)
    second = second - (second * p).sum(dim=(-2, -1), keepdim=True) / p_norm_sq * p
    tangent_norm = tangent.norm(dim=(-2, -1), keepdim=True)
    return torch.where(tangent_norm > 1e-12, tangent, second)


def angular_move_on_frobenius_sphere(p, direction, lr_t):
    p_norm = p.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    inner = (direction * p).sum(dim=(-2, -1), keepdim=True)
    tangent = direction - inner / p_norm.square() * p
    tangent_norm = tangent.norm(dim=(-2, -1), keepdim=True)
    fallback = angular_fallback_tangent(p)
    tangent = torch.where(tangent_norm > 1e-12, tangent, fallback)
    tangent_unit = tangent / tangent.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    lr = lr_t.to(dtype=p.dtype)
    rotated = p * torch.cos(lr) - tangent_unit * (p_norm * torch.sin(lr))
    return rotated / rotated.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12) * p_norm


@torch.compile(dynamic=False, fullgraph=True)
def angular_adamw_step_fused(
    p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t
):
    prev = p.clone()
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    direction = exp_avg / denom
    p.copy_(angular_move_on_frobenius_sphere(p, direction, lr_t))
    return prev - p


@torch.compile(dynamic=False, fullgraph=True)
def angular_muon_step_fused(
    stacked_grads,
    stacked_params,
    momentum_buffer,
    second_momentum_buffer,
    momentum_t,
    lr_t,
    beta2_t,
    ns_steps,
    red_dim,
):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
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
    prev = stacked_params.clone()
    stacked_params.copy_(angular_move_on_frobenius_sphere(stacked_params, g, lr_t))
    return prev - stacked_params


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

    def _step_angular_adamw(self, group, collect_update_norms=False):
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
            update = angular_adamw_step_fused(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
            )
            if collect_update_norms:
                p.grad = update.to(dtype=p.dtype)

    def _step_angular_muon(self, group, collect_update_norms=False):
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
        self._muon_lr_t.fill_(group["lr"])
        updates = angular_muon_step_fused(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
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
            elif group["kind"] == "angular_adamw":
                self._step_angular_adamw(group, collect_update_norms)
            elif group["kind"] == "angular_muon":
                self._step_angular_muon(group, collect_update_norms)


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
MAX_STEPS = 30  # exact number of optimizer steps to train
TOTAL_BATCH_SIZE = 2**17  # ~524K tokens per optimizer step

SCALAR_LR = 0.45
RANDOM_SEED = 42
EARLY_HIGH_LR_STEPS = 3
EARLY_HIGH_LR_KINDS = {
    "attn.c_proj",
    "attn.ve_gate",
    "k",
    "lm_head",
    "mlp.c_fc",
    "mlp.c_proj",
    "q",
    "v",
    "ve",
    "wte",
}
SCALE_LR_BY_WEIGHT_KIND = {
    "attn.c_proj": 0.2,
    "attn.ve_gate": 0.05,
    "k": 0.0,
    "lm_head": 0.2,
    "mlp.c_fc": 0.1,
    "mlp.c_proj": 0.2,
    "q": 0.0,
    "v": 0.05,
    "ve": 0.08,
    "wte": 0.03,
}
INITIAL_EFFECTIVE_LR_BY_WEIGHT_KIND = {
    "attn.c_proj": 0.144,
    "attn.ve_gate": 0.1913532,
    "k": 0.168,
    "lm_head": 0.48,
    "mlp.c_fc": 0.144,
    "mlp.c_proj": 0.144,
    "q": 0.168,
    "v": 0.072,
    "ve": 0.1986264,
    "wte": 0.375,
}

EARLY_HIGH_EFFECTIVE_LR_BY_WEIGHT_KIND = {
    "attn.c_proj": 0.108,
    "attn.ve_gate": 0.1435149,
    "k": 0.126,
    "lm_head": 0.08,
    "mlp.c_fc": 0.108,
    "mlp.c_proj": 0.108,
    "q": 0.126,
    "v": 0.054,
    "ve": 0.17,
    "wte": 0.07,
}

EFFECTIVE_LR_LOG_LINEAR_KNOTS = (
    (3, 1.0),
    (10, 1.25),
    (30, 0.3),
)

SCALE_LR_LOG_LINEAR_KNOTS = (
    (3, 1.0),
    (10, 1.25),
    (30, 0.3),
)

DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)
NORM_LOG_EVERY = 1  # optimizer steps between norm logs; 0 disables norm logs

# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

H100_BF16_PEAK_FLOPS = 989.5e12


def build_model_config(depth, vocab_size):
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


# LR scheduling:
# - log-scale params use the scale multiplier
# - angular matrix params use the effective multiplier, with a per-kind override
#   after EARLY_HIGH_LR_STEPS
# - scalar AdamW params stay at their initial LR


def log_linear_interpolate(step, knots):
    """Piecewise log-linear interpolation for positive schedule multipliers."""
    if not knots:
        raise ValueError("Schedule knots must not be empty.")

    if step <= knots[0][0]:
        return knots[0][1]

    for (left_step, left_value), (right_step, right_value) in zip(knots, knots[1:]):
        if step <= right_step:
            progress = (step - left_step) / (right_step - left_step)
            left_log = math.log(left_value)
            right_log = math.log(right_value)
            return math.exp(left_log + progress * (right_log - left_log))

    return knots[-1][1]


def get_effective_lr_multiplier(step):
    return log_linear_interpolate(step, EFFECTIVE_LR_LOG_LINEAR_KNOTS)


def get_scale_lr_multiplier(step):
    return log_linear_interpolate(step, SCALE_LR_LOG_LINEAR_KNOTS)


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def should_use_early_effective_lr(step, weight_kind):
    return (
        EARLY_HIGH_LR_STEPS > 0
        and step >= EARLY_HIGH_LR_STEPS
        and weight_kind in EARLY_HIGH_LR_KINDS
    )


def get_angular_group_lr(group, step, effective_lr_multiplier):
    weight_kind = group["weight_kind"]
    if should_use_early_effective_lr(step, weight_kind):
        base_lr = EARLY_HIGH_EFFECTIVE_LR_BY_WEIGHT_KIND[weight_kind]
    else:
        base_lr = group["initial_lr"]
    return base_lr * effective_lr_multiplier


def get_scheduled_group_lr(group, step, effective_lr_multiplier, scale_lr_multiplier):
    if group.get("scale_lr_schedule", False):
        return group["initial_lr"] * scale_lr_multiplier
    if group.get("constant_lr", False) or group["kind"] == "adamw":
        return group["initial_lr"]
    if group["kind"] in {"angular_muon", "angular_adamw"}:
        return get_angular_group_lr(group, step, effective_lr_multiplier)
    raise NotImplementedError(f"Unknown optimizer group kind: {group['kind']}")


def update_optimizer_hyperparams(optimizer, step):
    effective_lr_multiplier = get_effective_lr_multiplier(step)
    scale_lr_multiplier = get_scale_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)

    for group in optimizer.param_groups:
        group["lr"] = get_scheduled_group_lr(
            group, step, effective_lr_multiplier, scale_lr_multiplier
        )
        if group["kind"] == "angular_muon":
            group["momentum"] = muon_momentum

    return effective_lr_multiplier, scale_lr_multiplier


def matrix_scaled_norm(t):
    rows, cols = t.shape
    return t.detach().float().norm() / math.sqrt(max(rows, cols))


def scalar_abs(t):
    return t.detach().float().abs()


def log_scale_value(log_scale):
    return log_scale.detach().float().exp()


def log_float(value):
    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return value
    digits = 5 - 1 - math.floor(math.log10(abs(value)))
    rounded = round(value, digits)
    return 0.0 if rounded == 0 else rounded


def tensor_log_float(t):
    return log_float(t.item())


def vector_angle_rad(prev, current):
    prev = prev.detach().float().flatten()
    current = current.detach().float().flatten()
    prev_norm = prev.norm()
    current_norm = current.norm()
    if prev_norm == 0 or current_norm == 0:
        return None
    cosine = (prev.dot(current) / (prev_norm * current_norm)).clamp(-1, 1)
    return torch.acos(cosine)


def norm_multiplier(prev, current):
    prev_norm = prev.detach().float().norm()
    current_norm = current.detach().float().norm()
    if prev_norm == 0:
        return None
    return current_norm / prev_norm


def effective_scaled_matrix_norm(weight, log_scale):
    return matrix_scaled_norm(weight) * log_scale_value(log_scale)


@torch.no_grad()
def add_norm_pair(record, name, p, scalar_index=None):
    if scalar_index is None:
        weight_norm = matrix_scaled_norm(p)
        grad_norm = matrix_scaled_norm(p.grad) if p.grad is not None else None
    else:
        weight_norm = scalar_abs(p[scalar_index])
        grad_norm = scalar_abs(p.grad[scalar_index]) if p.grad is not None else None
    record["weight_norms"][name] = tensor_log_float(weight_norm)
    record["grad_norms"][name] = (
        None if grad_norm is None else tensor_log_float(grad_norm)
    )


@torch.no_grad()
def add_scaled_module_norm_pair(record, name, module):
    record["weight_norms"][name] = tensor_log_float(
        effective_scaled_matrix_norm(module.weight, module.log_scale)
    )
    record["grad_norms"][name] = (
        None
        if module.weight.grad is None
        else tensor_log_float(matrix_scaled_norm(module.weight.grad))
    )


@torch.no_grad()
def add_log_scale_pair(record, name, log_scale):
    record["weight_norms"][name] = tensor_log_float(log_scale_value(log_scale))
    record["grad_norms"][name] = (
        None
        if log_scale.grad is None
        else tensor_log_float(log_scale.grad.detach().float().abs())
    )


@torch.no_grad()
def add_update_norm_pair(record, name, p, scalar_index=None):
    if p.grad is None:
        record["update_norms"][name] = None
        record["update_scalar_multipliers"][name] = None
        if scalar_index is None:
            record["update_rotations_rad"][name] = None
        return
    if scalar_index is None:
        current = p.detach().float()
        update = p.grad.detach().float()
        prev = current + update
        update_norm = matrix_scaled_norm(update)
        rotation = vector_angle_rad(prev, current)
        scalar = norm_multiplier(prev, current)
        record["update_rotations_rad"][name] = (
            None if rotation is None else tensor_log_float(rotation)
        )
    else:
        current = p[scalar_index].detach().float()
        update = p.grad[scalar_index].detach().float()
        prev = current + update
        update_norm = scalar_abs(update)
        scalar = norm_multiplier(prev, current)
    record["update_norms"][name] = tensor_log_float(update_norm)
    record["update_scalar_multipliers"][name] = (
        None if scalar is None else tensor_log_float(scalar)
    )


@torch.no_grad()
def add_scaled_module_update_norm_pair(record, name, module):
    has_weight_update = module.weight.grad is not None
    has_scale_update = module.log_scale.grad is not None
    if not has_weight_update and not has_scale_update:
        record["update_norms"][name] = None
        record["update_scalar_multipliers"][name] = None
        record["update_rotations_rad"][name] = None
        return

    current_weight = module.weight.detach().float()
    weight_update = (
        module.weight.grad.detach().float()
        if has_weight_update
        else torch.zeros_like(current_weight)
    )
    prev_weight = current_weight + weight_update

    current_log_scale = module.log_scale.detach().float()
    log_scale_update = (
        module.log_scale.grad.detach().float()
        if has_scale_update
        else torch.zeros_like(current_log_scale)
    )
    prev_log_scale = current_log_scale + log_scale_update
    current_scale = current_log_scale.exp()
    prev_scale = prev_log_scale.exp()

    effective_update = prev_scale * prev_weight - current_scale * current_weight
    update_norm = matrix_scaled_norm(effective_update)
    rotation = vector_angle_rad(prev_weight, current_weight)
    scalar = norm_multiplier(prev_scale * prev_weight, current_scale * current_weight)

    record["update_norms"][name] = tensor_log_float(update_norm)
    record["update_rotations_rad"][name] = (
        None if rotation is None else tensor_log_float(rotation)
    )
    record["update_scalar_multipliers"][name] = (
        None if scalar is None else tensor_log_float(scalar)
    )


@torch.no_grad()
def add_log_scale_update_pair(record, name, log_scale):
    if log_scale.grad is None:
        record["update_norms"][name] = None
        record["update_scalar_multipliers"][name] = None
        return
    update = log_scale.grad.detach().float()
    current_scale = log_scale.detach().float().exp()
    prev_scale = (log_scale.detach().float() + update).exp()
    record["update_norms"][name] = tensor_log_float(
        (prev_scale - current_scale).abs()
    )
    record["update_scalar_multipliers"][name] = tensor_log_float(
        current_scale / prev_scale.clamp_min(1e-12)
    )


@torch.no_grad()
def build_norm_log(
    model,
    activation_norms,
    residual_mix_l2_fractions,
    residual_path_l2_fractions,
    step,
):
    record = {
        "type": "norms",
        "scaled_matrix_logging": "effective",
        "step": step,
        "weight_norms": {},
        "grad_norms": {},
        "update_norms": {},
        "update_rotations_rad": {},
        "update_scalar_multipliers": {},
        "activation_l2_norms": {},
        "residual_mix_l2_fractions": {},
        "residual_path_l2_fractions": {},
    }
    add_scaled_module_norm_pair(record, "wte", model.transformer.wte)
    add_scaled_module_norm_pair(record, "lm_head", model.lm_head)
    for name, _, module in model.scaled_matrix_module_specs():
        add_log_scale_pair(record, f"{name}.scale", module.log_scale)
    for i, block in enumerate(model.transformer.h):
        prefix = f"h.{i}"
        add_scaled_module_norm_pair(record, f"{prefix}.q", block.attn.c_q)
        add_scaled_module_norm_pair(record, f"{prefix}.k", block.attn.c_k)
        add_scaled_module_norm_pair(record, f"{prefix}.v", block.attn.c_v)
        add_scaled_module_norm_pair(
            record, f"{prefix}.attn.c_proj", block.attn.c_proj
        )
        add_scaled_module_norm_pair(record, f"{prefix}.mlp.c_fc", block.mlp.c_fc)
        add_scaled_module_norm_pair(record, f"{prefix}.mlp.c_proj", block.mlp.c_proj)
        add_norm_pair(record, f"{prefix}.resid_lambdas", model.resid_lambdas, i)
        add_norm_pair(record, f"{prefix}.x0_lambdas", model.x0_lambdas, i)
        if str(i) in model.value_embeds:
            add_scaled_module_norm_pair(
                record, f"{prefix}.ve", model.value_embeds[str(i)]
            )
        if block.attn.ve_gate is not None:
            add_scaled_module_norm_pair(
                record, f"{prefix}.attn.ve_gate", block.attn.ve_gate
            )
    if activation_norms is not None:
        if isinstance(activation_norms, dict):
            activation_items = activation_norms.items()
        else:
            activation_items = (
                (f"h.{i}", activation_norm)
                for i, activation_norm in enumerate(activation_norms)
            )
        for name, activation_norm in activation_items:
            record["activation_l2_norms"][name] = tensor_log_float(activation_norm)
    if residual_mix_l2_fractions is not None:
        for name, fraction in residual_mix_l2_fractions.items():
            record["residual_mix_l2_fractions"][name] = tensor_log_float(fraction)
    if residual_path_l2_fractions is not None:
        for name, fraction in residual_path_l2_fractions.items():
            record["residual_path_l2_fractions"][name] = tensor_log_float(fraction)
    return record


@torch.no_grad()
def add_update_norms(record, model):
    add_scaled_module_update_norm_pair(record, "wte", model.transformer.wte)
    add_scaled_module_update_norm_pair(record, "lm_head", model.lm_head)
    for name, _, module in model.scaled_matrix_module_specs():
        add_log_scale_update_pair(record, f"{name}.scale", module.log_scale)
    for i, block in enumerate(model.transformer.h):
        prefix = f"h.{i}"
        add_scaled_module_update_norm_pair(record, f"{prefix}.q", block.attn.c_q)
        add_scaled_module_update_norm_pair(record, f"{prefix}.k", block.attn.c_k)
        add_scaled_module_update_norm_pair(record, f"{prefix}.v", block.attn.c_v)
        add_scaled_module_update_norm_pair(
            record, f"{prefix}.attn.c_proj", block.attn.c_proj
        )
        add_scaled_module_update_norm_pair(
            record, f"{prefix}.mlp.c_fc", block.mlp.c_fc
        )
        add_scaled_module_update_norm_pair(
            record, f"{prefix}.mlp.c_proj", block.mlp.c_proj
        )
        add_update_norm_pair(
            record, f"{prefix}.resid_lambdas", model.resid_lambdas, i
        )
        add_update_norm_pair(record, f"{prefix}.x0_lambdas", model.x0_lambdas, i)
        if str(i) in model.value_embeds:
            add_scaled_module_update_norm_pair(
                record, f"{prefix}.ve", model.value_embeds[str(i)]
            )
        if block.attn.ve_gate is not None:
            add_scaled_module_update_norm_pair(
                record, f"{prefix}.attn.ve_gate", block.attn.ve_gate
            )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training():
    global fa3

    t_start = time.time()
    fa3 = initialize_flash_attention()
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

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
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()
    print(f"Estimated FLOPs per token: {num_flops_per_token:.4e}")

    tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    if TOTAL_BATCH_SIZE % tokens_per_fwdbwd != 0:
        raise ValueError(
            "TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE * MAX_SEQ_LEN "
            f"({TOTAL_BATCH_SIZE=} {tokens_per_fwdbwd=})."
        )
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)  # prefetch first batch

    uncompiled_model = model
    model = torch.compile(model, dynamic=False)
    optimizer = uncompiled_model.setup_optimizer(
        matrix_lrs=INITIAL_EFFECTIVE_LR_BY_WEIGHT_KIND,
        adam_betas=(0.8, 0.95),
        scalar_lr=SCALAR_LR,
    )

    print(f"Training steps: {MAX_STEPS}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    smooth_train_loss = 0.0
    total_training_time = 0.0
    train_loss_f = float("nan")
    debiased_smooth_loss = float("nan")
    step = 0

    while step < MAX_STEPS:
        torch.cuda.synchronize()
        t0 = time.time()
        should_log_norms = NORM_LOG_EVERY > 0 and step % NORM_LOG_EVERY == 0
        activation_norms = None
        residual_mix_l2_fractions = None
        residual_path_l2_fractions = None

        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                if should_log_norms and micro_step == grad_accum_steps - 1:
                    (
                        loss,
                        activation_norms,
                        residual_mix_l2_fractions,
                        residual_path_l2_fractions,
                    ) = model(x, y, return_activation_norms=True)
                else:
                    loss = model(x, y)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)

        effective_lr_multiplier, _ = update_optimizer_hyperparams(optimizer, step)
        if should_log_norms:
            norm_log_record = build_norm_log(
                uncompiled_model,
                activation_norms,
                residual_mix_l2_fractions,
                residual_path_l2_fractions,
                step,
            )
        else:
            norm_log_record = None

        optimizer.step(collect_update_norms=should_log_norms)
        if should_log_norms:
            add_update_norms(norm_log_record, uncompiled_model)
            print(f"\nNORM_LOG {json.dumps(norm_log_record, sort_keys=True)}", flush=True)
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        if not math.isfinite(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            sys.exit(1)

        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * min((step + 1) / MAX_STEPS, 1.0)
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
        remaining_steps = max(0, MAX_STEPS - step - 1)

        print(
            f"\rstep {step:05d} ({pct_done:.5g}%) | loss: {debiased_smooth_loss:.5g} | "
            f"lrm_non_scalar: {effective_lr_multiplier:.5g} | dt: {dt * 1000:.5g}ms | "
            f"tok/sec: {tok_per_sec:,} | mfu: {mfu:.5g}% | epoch: {epoch} | "
            f"remaining_steps: {remaining_steps}    ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

    print()  # newline after \r training log

    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

    total_tokens = step * TOTAL_BATCH_SIZE
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
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_bpb:          {val_bpb:.5g}")
    print(f"train_loss:       {train_loss_f:.5g}")
    print(f"smooth_train_loss:{debiased_smooth_loss:.5g}")
    print(f"training_seconds: {total_training_time:.5g}")
    print(f"total_seconds:    {t_end - t_start:.5g}")
    print(f"peak_vram_mb:     {peak_vram_mb:.5g}")
    print(f"mfu_percent:      {steady_state_mfu:.5g}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.5g}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.5g}")
    print(f"depth:            {DEPTH}")


def main():
    run_training()


if __name__ == "__main__":
    main()
