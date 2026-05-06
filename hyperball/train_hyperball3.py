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


def rms_norm(x):
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
        q, k = rms_norm(q), rms_norm(k)

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
            rms_norm(x), ve, cos_sin, window_size, activation_norms, prefix
        )
        log_residual_path_fractions(
            residual_path_l2_fractions, prefix, "attn", x, attn_out
        )
        x = x + attn_out
        mlp_out = self.mlp(rms_norm(x), activation_norms, prefix)
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
        weight_norm = {
            "h.0.attn.c_proj": 4.08,
            "h.0.k": 3.97,
            "h.0.mlp.c_fc": 3.75,
            "h.0.mlp.c_proj": 2.33,
            "h.0.q": 4.28,
            "h.0.v": 4.19,
            "h.1.attn.c_proj": 4.6,
            "h.1.attn.ve_gate": 0.341,
            "h.1.k": 4.17,
            "h.1.mlp.c_fc": 3.86,
            "h.1.mlp.c_proj": 2.37,
            "h.1.q": 4.31,
            "h.1.v": 5.21,
            "h.1.ve": 231.0,
            "h.2.attn.c_proj": 4.41,
            "h.2.k": 4.01,
            "h.2.mlp.c_fc": 3.75,
            "h.2.mlp.c_proj": 2.4,
            "h.2.q": 4.09,
            "h.2.v": 4.51,
            "h.3.attn.c_proj": 5.13,
            "h.3.attn.ve_gate": 0.635,
            "h.3.k": 3.74,
            "h.3.mlp.c_fc": 3.69,
            "h.3.mlp.c_proj": 2.31,
            "h.3.q": 3.62,
            "h.3.v": 5.2,
            "h.3.ve": 231.0,
            "h.4.attn.c_proj": 4.77,
            "h.4.k": 4.2,
            "h.4.mlp.c_fc": 3.88,
            "h.4.mlp.c_proj": 2.49,
            "h.4.q": 4.21,
            "h.4.v": 5.13,
            "h.5.attn.c_proj": 4.64,
            "h.5.attn.ve_gate": 0.83,
            "h.5.k": 4.44,
            "h.5.mlp.c_fc": 3.93,
            "h.5.mlp.c_proj": 2.57,
            "h.5.q": 4.41,
            "h.5.v": 5.35,
            "h.5.ve": 236.0,
            "h.6.attn.c_proj": 4.75,
            "h.6.k": 4.13,
            "h.6.mlp.c_fc": 3.91,
            "h.6.mlp.c_proj": 2.56,
            "h.6.q": 4.12,
            "h.6.v": 5.2,
            "h.7.attn.c_proj": 5.33,
            "h.7.attn.ve_gate": 1.68,
            "h.7.k": 3.96,
            "h.7.mlp.c_fc": 3.67,
            "h.7.mlp.c_proj": 2.52,
            "h.7.q": 3.93,
            "h.7.v": 4.79,
            "h.7.ve": 242.0,
            "lm_head": 3.57,
            "wte": 406.0,
        }

        def actual_frobenius_norm(w, norm_value):
            return norm_value * (max(w.shape) ** 0.5)

        def apply_target_norm(w, norm_value):
            if NORM_SCHEME == "matrix":
                w.div_(w.norm().clamp_min(1e-12)).mul_(norm_value)
                return
            output_dim, input_dim = w.shape
            if NORM_SCHEME == "per_output" or (
                NORM_SCHEME == "per_smaller_vector" and output_dim >= input_dim
            ):
                norm_value = norm_value / (output_dim**0.5)
                w.div_(w.norm(dim=1, keepdim=True).clamp_min(1e-12)).mul_(norm_value)
            else:
                norm_value = norm_value / (input_dim**0.5)
                w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-12)).mul_(norm_value)

        def factor_scaled_matrix(module):
            norm = module.weight.detach().float().norm().clamp_min(1e-12)
            module.weight.div_(norm.to(dtype=module.weight.dtype))
            module.log_scale.fill_(math.log(float(norm)))

        def init_normal_scaled(module, mean=0.0, std=1.0, target_norm=None):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)
            if target_norm is not None:
                apply_target_norm(module.weight, target_norm)
            factor_scaled_matrix(module)

        def init_uniform_scaled(module, low, high, target_norm=None):
            torch.nn.init.uniform_(module.weight, low, high)
            if target_norm is not None:
                apply_target_norm(module.weight, target_norm)
            factor_scaled_matrix(module)

        def init_scaled_projection(module):
            if PROJECTION_INIT_MODE == "identity":
                module.weight.zero_()
                rows, cols = module.weight.shape
                diag = min(rows, cols)
                module.weight[:diag, :diag].fill_diagonal_(1.0)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
            module.weight.div_(module.weight.norm().clamp_min(1e-12))
            module.log_scale.fill_(math.log(PROJECTION_INIT_SCALE))

        init_normal_scaled(self.transformer.wte, mean=0.0, std=1.0)
        init_normal_scaled(self.lm_head, mean=0.0, std=0.001)

        for layer_idx, block in enumerate(self.transformer.h):
            prefix = f"h.{layer_idx}"
            matrix_modules = (
                (block.attn.c_q, f"{prefix}.q"),
                (block.attn.c_k, f"{prefix}.k"),
                (block.attn.c_v, f"{prefix}.v"),
            )
            for module, key in matrix_modules:
                init_normal_scaled(
                    module,
                    mean=0.0,
                    std=1.0,
                    target_norm=actual_frobenius_norm(module.weight, weight_norm[key]),
                )
            init_scaled_projection(block.attn.c_proj)
            s = (2 / block.mlp.c_fc.weight.shape[1]) ** 0.5
            init_uniform_scaled(block.mlp.c_fc, -s, s)
            init_scaled_projection(block.mlp.c_proj)
            if block.attn.ve_gate is not None:
                key = f"{prefix}.attn.ve_gate"
                init_normal_scaled(
                    block.attn.ve_gate,
                    mean=0.0,
                    std=1.0,
                    target_norm=actual_frobenius_norm(
                        block.attn.ve_gate.weight, weight_norm[key]
                    ),
                )

        for layer_idx, ve in self.value_embeds.items():
            init_normal_scaled(
                ve,
                mean=0.0,
                std=1.0,
                target_norm=actual_frobenius_norm(
                    ve.weight, weight_norm[f"h.{layer_idx}.ve"]
                ),
            )

        # Per-layer scalars
        self.resid_lambdas.fill_(RESID_LAMBDA_INIT)
        self.x0_lambdas.fill_(X0_LAMBDA_INIT)
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

    def setup_optimizer(
        self,
        matrix_lrs=None,
        matrix_weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
    ):
        if matrix_lrs is None:
            matrix_lrs = FITTED_EFFECTIVE_LR_BY_WEIGHT_KIND
        angular_muon_param_specs = []
        for block in self.transformer.h:
            angular_muon_param_specs.extend(
                [
                    ("q", block.attn.c_q.weight),
                    ("k", block.attn.c_k.weight),
                    ("v", block.attn.c_v.weight),
                    ("attn.c_proj", block.attn.c_proj.weight),
                    ("mlp.c_fc", block.mlp.c_fc.weight),
                    ("mlp.c_proj", block.mlp.c_proj.weight),
                ]
            )
            if block.attn.ve_gate is not None:
                angular_muon_param_specs.append(
                    ("attn.ve_gate", block.attn.ve_gate.weight)
                )
        angular_adamw_param_specs = [
            ("wte", self.transformer.wte.weight),
            ("lm_head", self.lm_head.weight),
        ]
        angular_adamw_param_specs.extend(
            ("ve", ve.weight) for ve in self.value_embeds.values()
        )
        scaled_log_scale_param_specs = [
            (weight_kind, module.log_scale)
            for _, weight_kind, module in self.scaled_matrix_module_specs()
        ]
        angular_muon_params = [p for _, p in angular_muon_param_specs]
        angular_adamw_params = [p for _, p in angular_adamw_param_specs]
        scaled_log_scale_params = [p for _, p in scaled_log_scale_param_specs]
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(angular_muon_params)
            + len(angular_adamw_params)
            + len(scaled_log_scale_params)
            + len(resid_params)
            + len(x0_params)
        )
        print(
            "Using fixed-norm angular directions for every 2D matrix "
            f"with projection_init_scale={PROJECTION_INIT_SCALE:g}"
        )
        param_groups = [
            dict(
                kind="adamw",
                params=resid_params,
                lr=scalar_lr * 0.01,  # TODO.
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
        for weight_kind in sorted({kind for kind, _ in scaled_log_scale_param_specs}):
            group_params = [
                param
                for param_kind, param in scaled_log_scale_param_specs
                if param_kind == weight_kind
            ]
            param_groups.append(
                dict(
                    kind="adamw",
                    weight_kind=f"{weight_kind}.scale",
                    params=group_params,
                    lr=SCALE_LR_BY_WEIGHT_KIND[weight_kind],
                    betas=SCALE_ADAM_BETAS,
                    eps=1e-10,
                    weight_decay=0.0,
                    constant_lr=True,
                )
            )
        angular_muon_group_keys = sorted(
            {
                (weight_kind, tuple(param.shape))
                for weight_kind, param in angular_muon_param_specs
            }
        )
        for weight_kind, shape in angular_muon_group_keys:
            group_params = [
                param
                for param_kind, param in angular_muon_param_specs
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
        for weight_kind, param in angular_adamw_param_specs:
            param_groups.append(
                dict(
                    kind="angular_adamw",
                    weight_kind=weight_kind,
                    params=[param],
                    lr=matrix_lrs[weight_kind],
                    betas=adam_betas,
                    eps=1e-10,
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
        x = rms_norm(x)
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
        x = rms_norm(x)

        softcap = LOGIT_SOFTCAP
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


def adamh_norm(w):
    if NORM_SCHEME == "matrix":
        return w.norm()
    output_dim, input_dim = w.shape
    if NORM_SCHEME == "per_output" or (
        NORM_SCHEME == "per_smaller_vector" and output_dim >= input_dim
    ):
        return w.norm(dim=1, keepdim=True)
    else:
        return w.norm(dim=0, keepdim=True)


@torch.compile(dynamic=False, fullgraph=True)
def adamh_step_fused(
    p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t
):
    # p.mul_(1 - lr_t * wd_t)
    prev = p.clone()
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t**step_t
    # the denom bias ajustment need to account for eps,
    # but the numerator bias adjustment can be removed
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    # not needed after norm
    # exp_avg = exp_avg / bias1
    update = exp_avg / denom
    update_norm = adamh_norm(update)
    p_norm = adamh_norm(p)
    update = update / update_norm.clamp_min(1e-12) * p_norm * lr_t
    p.add_(update, alpha=-1)
    p.div_(adamh_norm(p).clamp_min(1e-12)).mul_(p_norm)
    return prev - p


# TODO: this one should be usable for adamh_norm as well.
def muon_norm(w):
    if NORM_SCHEME == "matrix":
        return w.norm(dim=(-2, -1), keepdim=True)
    output_dim, input_dim = w.shape[-2:]
    if NORM_SCHEME == "per_output" or (
        NORM_SCHEME == "per_smaller_vector" and output_dim >= input_dim
    ):
        return w.norm(dim=-1, keepdim=True)
    else:
        return w.norm(dim=-2, keepdim=True)


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
    # putting the wd into the update vector cuz cautious wd might be doing more than keeping the weight norm.
    # update = g + wd * stacked_params * mask
    prev = stacked_params.clone()
    update = g

    update_norm = muon_norm(update)
    p_norm = muon_norm(stacked_params)
    update = update / update_norm.clamp_min(1e-12) * p_norm * lr
    stacked_params.sub_(update)
    stacked_params.div_(muon_norm(stacked_params).clamp_min(1e-12)).mul_(p_norm)
    return prev - stacked_params


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


@torch.compile(dynamic=False, fullgraph=True)
def muon_legacy_step_fused(
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

    def _step_adamh(self, group, collect_update_norms=False):
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
            update = adamh_step_fused(
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
        # this lr scaling is not needed cuz of hyperball
        # group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        self._muon_lr_t.fill_(group["lr"])
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

    def _step_muon_legacy(self, group, collect_update_norms=False):
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
        updates = muon_legacy_step_fused(
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
        for inner_step in range(OPTIMIZER_INNER_STEPS):
            collect_this_step = collect_update_norms and inner_step == OPTIMIZER_INNER_STEPS - 1
            for group in self.param_groups:
                if group["kind"] == "adamw":
                    self._step_adamw(group, collect_this_step)
                elif group["kind"] == "angular_adamw":
                    self._step_angular_adamw(group, collect_this_step)
                elif group["kind"] == "adamh":
                    self._step_adamh(group, collect_this_step)
                elif group["kind"] == "muon":
                    self._step_muon(group, collect_this_step)
                elif group["kind"] == "angular_muon":
                    self._step_angular_muon(group, collect_this_step)
                elif group["kind"] == "muon_legacy":
                    self._step_muon_legacy(group, collect_this_step)


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
SCHEDULE_STEPS = 1350  # keep the fitted long-horizon LR schedule for 30-step runs
TOTAL_BATCH_SIZE = 2**17  # ~524K tokens per optimizer step
SCALAR_LR = 0.75  # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.1  # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95)  # Adam beta1, beta2
SCALE_ADAM_BETAS = (0.0, 0.95)


def env_float(name, default):
    return float(os.environ.get(name, default))


PROJECTION_INIT_SCALE = env_float("HYPERBALL_PROJECTION_INIT_SCALE", 5.0)
PROJECTION_INIT_MODE = os.environ.get("HYPERBALL_PROJECTION_INIT_MODE", "normal")
DATA_DEPENDENT_PROJECTION_INIT = bool(
    int(os.environ.get("HYPERBALL_DATA_DEPENDENT_PROJECTION_INIT", "0"))
)
DATA_DEPENDENT_UNIGRAM_INIT = bool(
    int(os.environ.get("HYPERBALL_DATA_DEPENDENT_UNIGRAM_INIT", "0"))
)
UNIGRAM_INIT_ALPHA = env_float("HYPERBALL_UNIGRAM_INIT_ALPHA", 1.0)
UNIGRAM_INIT_COMMON = env_float("HYPERBALL_UNIGRAM_INIT_COMMON", 6.0)
DATA_DEPENDENT_RIDGE_HEAD_INIT = bool(
    int(os.environ.get("HYPERBALL_DATA_DEPENDENT_RIDGE_HEAD_INIT", "0"))
)
RIDGE_HEAD_LAMBDA = env_float("HYPERBALL_RIDGE_HEAD_LAMBDA", 0.01)
RIDGE_HEAD_LOGIT_SCALE = env_float("HYPERBALL_RIDGE_HEAD_LOGIT_SCALE", 30.0)
PROJECTION_WARMSTART_SCALE = env_float("HYPERBALL_PROJECTION_WARMSTART_SCALE", 1.0)
RESID_LAMBDA_INIT = env_float("HYPERBALL_RESID_LAMBDA_INIT", 1.0)
X0_LAMBDA_INIT = env_float("HYPERBALL_X0_LAMBDA_INIT", 5.0)
TRAIN_LOSS_EMA_BETA = 0.9
LOGIT_SOFTCAP = env_float("HYPERBALL_LOGIT_SOFTCAP", 15.0)
OPTIMIZER_INNER_STEPS = int(os.environ.get("HYPERBALL_OPTIMIZER_INNER_STEPS", "1"))
POST_UPDATE_TRAIN_LOSS = bool(
    int(os.environ.get("HYPERBALL_POST_UPDATE_TRAIN_LOSS", "1"))
)
SCALE_LR_BY_WEIGHT_KIND = {
    "attn.c_proj": env_float("HYPERBALL_SCALE_LR_ATTN_C_PROJ", 0.2),
    "attn.ve_gate": env_float("HYPERBALL_SCALE_LR_ATTN_VE_GATE", 0.05),
    "k": 0.0,
    "lm_head": env_float("HYPERBALL_SCALE_LR_LM_HEAD", 0.3),
    "mlp.c_fc": env_float("HYPERBALL_SCALE_LR_MLP_C_FC", 0.1),
    "mlp.c_proj": env_float("HYPERBALL_SCALE_LR_MLP_C_PROJ", 0.2),
    "q": 0.0,
    "v": env_float("HYPERBALL_SCALE_LR_V", 0.05),
    "ve": env_float("HYPERBALL_SCALE_LR_VE", 0.05),
    "wte": 0.0,
}
FITTED_EFFECTIVE_LR_BY_WEIGHT_KIND = {
    "attn.c_proj": env_float("HYPERBALL_LR_ATTN_C_PROJ", 0.08),
    "attn.ve_gate": env_float("HYPERBALL_LR_ATTN_VE_GATE", 0.106307),
    "k": env_float("HYPERBALL_LR_K", 0.0636095),
    "lm_head": env_float("HYPERBALL_LR_LM_HEAD", 0.055),
    "mlp.c_fc": env_float("HYPERBALL_LR_MLP_C_FC", 0.0659059),
    "mlp.c_proj": env_float("HYPERBALL_LR_MLP_C_PROJ", 0.08),
    "q": env_float("HYPERBALL_LR_Q", 0.0628415),
    "v": env_float("HYPERBALL_LR_V", 0.0491089),
    "ve": env_float("HYPERBALL_LR_VE", 0.110348),
    "wte": env_float("HYPERBALL_LR_WTE", 0.047),
}
EFFECTIVE_LR_LOG_LINEAR_KNOTS = (
    (20, 1.0),
    (200, 0.458328),
    (400, 0.282636),
    (700, 0.170964),
    (1000, 0.0958409),
    (1200, 0.0477203),
    (1349, 0.0118781),
)
OLD_TRAIN_WARMUP_RATIO = 0.0
OLD_TRAIN_WARMDOWN_RATIO = 0.7
OLD_TRAIN_FINAL_LR_FRAC = 0.05
NORM_SCHEME = "per_smaller_vector"
assert NORM_SCHEME in {
    "matrix",
    "per_output",
    "per_input",
    "per_smaller_vector",
}

DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)
NORM_LOG_EVERY = 0  # optimizer steps between norm logs; 0 disables norm logs


@torch.no_grad()
def projection_modules(model):
    for block in model.transformer.h:
        yield block.attn.c_proj
        yield block.mlp.c_proj


def warmstart_projection_directions(forward_model, param_model, x, y):
    if not DATA_DEPENDENT_PROJECTION_INIT:
        return
    param_model.zero_grad(set_to_none=True)
    with autocast_ctx:
        loss = forward_model(x, y)
    loss.backward()
    with torch.no_grad():
        for module in projection_modules(param_model):
            if module.weight.grad is None:
                continue
            direction = -module.weight.grad.detach().float()
            direction_norm = direction.norm()
            if direction_norm <= 0:
                continue
            module.weight.copy_(
                (direction / direction_norm).to(dtype=module.weight.dtype)
            )
            module.log_scale.fill_(math.log(PROJECTION_WARMSTART_SCALE))
    param_model.zero_grad(set_to_none=True)
    print(
        "Data-dependent projection warmstart: "
        f"scale={PROJECTION_WARMSTART_SCALE:g}, loss={loss.item():.5g}"
    )


@torch.no_grad()
def refactor_scaled_weight(module, effective_weight):
    norm = effective_weight.float().norm().clamp_min(1e-12)
    module.weight.copy_((effective_weight / norm).to(dtype=module.weight.dtype))
    module.log_scale.fill_(math.log(float(norm)))


@torch.no_grad()
def forward_hidden(model, idx):
    B, T = idx.size()
    cos_sin = model.cos[:, :T], model.sin[:, :T]
    x = model.transformer.wte(idx)
    x = rms_norm(x)
    x0 = x
    for i, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
        ve = model.value_embeds[str(i)](idx) if str(i) in model.value_embeds else None
        x = block(x, ve, cos_sin, model.window_sizes[i])
    return rms_norm(x)


@torch.no_grad()
def warmstart_unigram_output_prior(param_model, x, y):
    if not DATA_DEPENDENT_UNIGRAM_INIT:
        return
    labels = y.reshape(-1)
    labels = labels[labels >= 0]
    counts = torch.bincount(labels, minlength=param_model.config.vocab_size).float()
    counts = counts.to(device=param_model.lm_head.weight.device)
    probs = (counts + UNIGRAM_INIT_ALPHA) / (
        counts.sum() + UNIGRAM_INIT_ALPHA * counts.numel()
    )
    logits = probs.log()
    logits = logits - logits.mean()

    wte = param_model.transformer.wte
    current_wte = wte.weight.detach().float() * wte.log_scale.detach().float().exp()
    common = torch.zeros(param_model.config.n_embd, device=current_wte.device)
    common[0] = UNIGRAM_INIT_COMMON
    effective_wte = current_wte + common[None, :]
    refactor_scaled_weight(wte, effective_wte)

    with autocast_ctx:
        hidden = forward_hidden(param_model, x)
    common_dot = hidden[..., 0].float().mean().abs().clamp_min(1e-3)
    lm_head = param_model.lm_head
    effective_lm_head = (
        lm_head.weight.detach().float() * lm_head.log_scale.detach().float().exp()
    )
    effective_lm_head[:, 0] += logits / common_dot
    refactor_scaled_weight(param_model.lm_head, effective_lm_head)
    print(
        "Data-dependent unigram output init: "
        f"alpha={UNIGRAM_INIT_ALPHA:g}, common={UNIGRAM_INIT_COMMON:g}, "
        f"entropy={float(-(probs * probs.log()).sum()):.5g}"
    )


@torch.no_grad()
def warmstart_ridge_lm_head(param_model, x, y):
    if not DATA_DEPENDENT_RIDGE_HEAD_INIT:
        return
    with autocast_ctx:
        hidden = forward_hidden(param_model, x)
    hidden = hidden.float().reshape(-1, param_model.config.n_embd)
    labels = y.reshape(-1)
    valid = labels >= 0
    hidden = hidden[valid]
    labels = labels[valid].to(device=hidden.device, dtype=torch.long)
    hidden = hidden - hidden.mean(dim=0, keepdim=True)
    n = hidden.size(0)
    gram = hidden.T @ hidden / n
    gram.diagonal().add_(RIDGE_HEAD_LAMBDA)
    rhs = torch.zeros(
        param_model.config.n_embd,
        param_model.config.vocab_size,
        device=hidden.device,
        dtype=torch.float32,
    )
    rhs.index_add_(1, labels, hidden.T)
    rhs.div_(n)
    effective_lm_head = torch.linalg.solve(gram, rhs).T.contiguous()
    effective_lm_head.mul_(RIDGE_HEAD_LOGIT_SCALE)
    refactor_scaled_weight(param_model.lm_head, effective_lm_head)
    print(
        "Data-dependent ridge lm_head init: "
        f"lambda={RIDGE_HEAD_LAMBDA:g}, logit_scale={RIDGE_HEAD_LOGIT_SCALE:g}"
    )


# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


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


config = build_model_config(DEPTH)
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
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch
uncompiled_model = model
warmstart_unigram_output_prior(uncompiled_model, x, y)
warmstart_ridge_lm_head(uncompiled_model, x, y)
model = torch.compile(model, dynamic=False)
warmstart_projection_directions(model, uncompiled_model, x, y)

optimizer = uncompiled_model.setup_optimizer(
    matrix_lrs=FITTED_EFFECTIVE_LR_BY_WEIGHT_KIND,
    matrix_weight_decay=WEIGHT_DECAY,
    adam_betas=ADAM_BETAS,
    scalar_lr=SCALAR_LR,
)

print(f"Training steps: {MAX_STEPS}")
print(f"Schedule steps: {SCHEDULE_STEPS}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on fixed-step progress)


def get_lr_multiplier_old_train(progress):
    if progress < OLD_TRAIN_WARMUP_RATIO:
        return progress / OLD_TRAIN_WARMUP_RATIO if OLD_TRAIN_WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - OLD_TRAIN_WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / OLD_TRAIN_WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * OLD_TRAIN_FINAL_LR_FRAC


def get_lr_multiplier_non_scalar(steps):
    knots = EFFECTIVE_LR_LOG_LINEAR_KNOTS
    if steps <= knots[0][0]:
        return knots[0][1]
    for (left_step, left_value), (right_step, right_value) in zip(knots, knots[1:]):
        if steps <= right_step:
            frac = (steps - left_step) / (right_step - left_step)
            left_log = math.log(left_value)
            right_log = math.log(right_value)
            return math.exp(left_log + frac * (right_log - left_log))
    return knots[-1][1]


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)


def get_weight_decay_old_train(progress):
    return WEIGHT_DECAY * (1 - progress)


def matrix_scaled_norm(t):
    rows, cols = t.shape
    return t.detach().float().norm() / math.sqrt(max(rows, cols))


def scalar_abs(t):
    return t.detach().float().abs()


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
def add_log_scale_pair(record, name, log_scale):
    record["weight_norms"][name] = tensor_log_float(log_scale.detach().float().exp())
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
def add_log_scale_update_pair(record, name, log_scale):
    if log_scale.grad is None:
        record["update_norms"][name] = None
        record["update_scalar_multipliers"][name] = None
        return
    update = log_scale.grad.detach().float()
    record["update_norms"][name] = tensor_log_float(update.abs())
    record["update_scalar_multipliers"][name] = tensor_log_float((-update).exp())


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
    add_norm_pair(record, "wte", model.transformer.wte.weight)
    add_norm_pair(record, "lm_head", model.lm_head.weight)
    for name, _, module in model.scaled_matrix_module_specs():
        add_log_scale_pair(record, f"{name}.scale", module.log_scale)
    for i, block in enumerate(model.transformer.h):
        prefix = f"h.{i}"
        add_norm_pair(record, f"{prefix}.q", block.attn.c_q.weight)
        add_norm_pair(record, f"{prefix}.k", block.attn.c_k.weight)
        add_norm_pair(record, f"{prefix}.v", block.attn.c_v.weight)
        add_norm_pair(record, f"{prefix}.attn.c_proj", block.attn.c_proj.weight)
        add_norm_pair(record, f"{prefix}.mlp.c_fc", block.mlp.c_fc.weight)
        add_norm_pair(record, f"{prefix}.mlp.c_proj", block.mlp.c_proj.weight)
        add_norm_pair(record, f"{prefix}.resid_lambdas", model.resid_lambdas, i)
        add_norm_pair(record, f"{prefix}.x0_lambdas", model.x0_lambdas, i)
        if str(i) in model.value_embeds:
            add_norm_pair(record, f"{prefix}.ve", model.value_embeds[str(i)].weight)
        if block.attn.ve_gate is not None:
            add_norm_pair(record, f"{prefix}.attn.ve_gate", block.attn.ve_gate.weight)
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
    add_update_norm_pair(record, "wte", model.transformer.wte.weight)
    add_update_norm_pair(record, "lm_head", model.lm_head.weight)
    for name, _, module in model.scaled_matrix_module_specs():
        add_log_scale_update_pair(record, f"{name}.scale", module.log_scale)
    for i, block in enumerate(model.transformer.h):
        prefix = f"h.{i}"
        add_update_norm_pair(record, f"{prefix}.q", block.attn.c_q.weight)
        add_update_norm_pair(record, f"{prefix}.k", block.attn.c_k.weight)
        add_update_norm_pair(record, f"{prefix}.v", block.attn.c_v.weight)
        add_update_norm_pair(record, f"{prefix}.attn.c_proj", block.attn.c_proj.weight)
        add_update_norm_pair(record, f"{prefix}.mlp.c_fc", block.mlp.c_fc.weight)
        add_update_norm_pair(record, f"{prefix}.mlp.c_proj", block.mlp.c_proj.weight)
        add_update_norm_pair(record, f"{prefix}.resid_lambdas", model.resid_lambdas, i)
        add_update_norm_pair(record, f"{prefix}.x0_lambdas", model.x0_lambdas, i)
        if str(i) in model.value_embeds:
            add_update_norm_pair(
                record, f"{prefix}.ve", model.value_embeds[str(i)].weight
            )
        if block.attn.ve_gate is not None:
            add_update_norm_pair(
                record, f"{prefix}.attn.ve_gate", block.attn.ve_gate.weight
            )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    should_log_norms = NORM_LOG_EVERY > 0 and step % NORM_LOG_EVERY == 0
    activation_norms = None
    residual_mix_l2_fractions = None
    residual_path_l2_fractions = None
    train_loss_x = None
    train_loss_y = None
    for micro_step in range(grad_accum_steps):
        train_loss_x = x
        train_loss_y = y
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
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(step / max(1, SCHEDULE_STEPS - 1), 1.0)
    lrm_non_scalar = get_lr_multiplier_non_scalar(step)
    lrm_old_train = get_lr_multiplier_old_train(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    muon_weight_decay_old_train = get_weight_decay_old_train(progress)
    for group in optimizer.param_groups:
        if group.get("constant_lr", False):
            group["lr"] = group["initial_lr"]
        elif group["kind"] in {
            "muon",
            "adamh",
            "angular_muon",
            "angular_adamw",
        }:
            group["lr"] = group["initial_lr"] * lrm_non_scalar
        elif group["kind"] == "muon_legacy":
            group["lr"] = group["initial_lr"] * lrm_old_train
        elif group["kind"] == "adamw":
            group["lr"] = group["initial_lr"] * lrm_old_train
        else:
            raise NotImplementedError()
        if group["kind"] in {"muon", "angular_muon"}:
            group["momentum"] = muon_momentum
            if group["kind"] == "muon":
                group["weight_decay"] = muon_weight_decay
        elif group["kind"] == "muon_legacy":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay_old_train
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
    if POST_UPDATE_TRAIN_LOSS:
        with torch.no_grad(), autocast_ctx:
            train_loss = model(train_loss_x, train_loss_y).detach()
    if should_log_norms:
        add_update_norms(norm_log_record, uncompiled_model)
        print(f"\nNORM_LOG {json.dumps(norm_log_record, sort_keys=True)}", flush=True)
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = TRAIN_LOSS_EMA_BETA
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * min((step + 1) / MAX_STEPS, 1.0)
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining_steps = max(0, MAX_STEPS - step - 1)

    print(
        f"\rstep {step:05d} ({pct_done:.5g}%) | loss: {debiased_smooth_loss:.5g} | lrm_non_scalar: {lrm_non_scalar:.5g} | lrm_old_train: {lrm_old_train:.5g} | dt: {dt * 1000:.5g}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.5g}% | epoch: {epoch} | remaining_steps: {remaining_steps}    ",
        end="",
        flush=True,
    )

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if step >= MAX_STEPS:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
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
print(f"smooth_train_loss:{min(debiased_smooth_loss, train_loss_f):.5g}")
print(f"training_seconds: {total_training_time:.5g}")
print(f"total_seconds:    {t_end - t_start:.5g}")
print(f"peak_vram_mb:     {peak_vram_mb:.5g}")
print(f"mfu_percent:      {steady_state_mfu:.5g}")
print(f"total_tokens_M:   {total_tokens / 1e6:.5g}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.5g}")
print(f"depth:            {DEPTH}")
