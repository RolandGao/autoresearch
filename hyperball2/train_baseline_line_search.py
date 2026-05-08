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
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


LINE_SEARCH_EXPERIMENTS = [
    {
        "name": "one_step_heldout",
        "description": "one-step global LR search on one held-out batch",
        "env": {
            "LINE_SEARCH_INTERVAL": "1",
            "LINE_SEARCH_HORIZON_STEPS": "1",
            "LINE_SEARCH_HELD_OUT_BATCHES": "1",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "1",
            "LINE_SEARCH_SAFE_LOSS_ATOL": "0",
            "LINE_SEARCH_SAFE_LOSS_RTOL": "0",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1000",
            "LINE_SEARCH_LOG_LR_EMA": "1",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rotating_safe",
        "description": "largest-safe global LR on rotating held-out batches",
        "env": {
            "LINE_SEARCH_INTERVAL": "1",
            "LINE_SEARCH_HORIZON_STEPS": "1",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1000",
            "LINE_SEARCH_LOG_LR_EMA": "1",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rotating_smooth",
        "description": "one-step rotating held-out search with LR inertia",
        "env": {
            "LINE_SEARCH_INTERVAL": "1",
            "LINE_SEARCH_HORIZON_STEPS": "1",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1.08",
            "LINE_SEARCH_LOG_LR_EMA": "0.35",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "next_batch_safe",
        "description": "one-step largest-safe global LR on future train batches",
        "env": {
            "LINE_SEARCH_INTERVAL": "1",
            "LINE_SEARCH_HORIZON_STEPS": "1",
            "LINE_SEARCH_VAL_SET_NAME": "next_train_batch",
            "LINE_SEARCH_HELD_OUT_BATCHES": "1",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1000",
            "LINE_SEARCH_LOG_LR_EMA": "1",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout2_safe",
        "description": "2-step rollout with largest-safe global LR",
        "env": {
            "LINE_SEARCH_INTERVAL": "4",
            "LINE_SEARCH_HORIZON_STEPS": "2",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1000",
            "LINE_SEARCH_LOG_LR_EMA": "1",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout2_smooth",
        "description": "2-step rollout with progress score and LR inertia",
        "env": {
            "LINE_SEARCH_INTERVAL": "4",
            "LINE_SEARCH_HORIZON_STEPS": "2",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0.05",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1.08",
            "LINE_SEARCH_LOG_LR_EMA": "0.35",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout4_safe",
        "description": "4-step rollout with largest-safe global LR",
        "env": {
            "LINE_SEARCH_INTERVAL": "8",
            "LINE_SEARCH_HORIZON_STEPS": "4",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1000",
            "LINE_SEARCH_LOG_LR_EMA": "1",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout4_smooth",
        "description": "4-step rollout with progress score and LR inertia",
        "env": {
            "LINE_SEARCH_INTERVAL": "8",
            "LINE_SEARCH_HORIZON_STEPS": "4",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0.05",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1.08",
            "LINE_SEARCH_LOG_LR_EMA": "0.35",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout4_smooth_norm",
        "description": "4-step rollout with progress score, inertia, and update-norm feedback",
        "env": {
            "LINE_SEARCH_INTERVAL": "8",
            "LINE_SEARCH_HORIZON_STEPS": "4",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0.05",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1.08",
            "LINE_SEARCH_LOG_LR_EMA": "0.35",
            "LINE_SEARCH_NORM_FEEDBACK": "1",
        },
    },
    {
        "name": "rollout4_strict",
        "description": "4-step rollout with stricter largest-safe tolerance",
        "env": {
            "LINE_SEARCH_INTERVAL": "8",
            "LINE_SEARCH_HORIZON_STEPS": "4",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SAFE_LOSS_ATOL": "1e-4",
            "LINE_SEARCH_SAFE_LOSS_RTOL": "5e-5",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0.05",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1.08",
            "LINE_SEARCH_LOG_LR_EMA": "0.35",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout4_loose",
        "description": "4-step rollout with looser largest-safe tolerance",
        "env": {
            "LINE_SEARCH_INTERVAL": "8",
            "LINE_SEARCH_HORIZON_STEPS": "4",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SAFE_LOSS_ATOL": "1e-3",
            "LINE_SEARCH_SAFE_LOSS_RTOL": "3e-4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0.05",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1.08",
            "LINE_SEARCH_LOG_LR_EMA": "0.35",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout8_safe",
        "description": "8-step rollout with largest-safe global LR",
        "env": {
            "LINE_SEARCH_INTERVAL": "16",
            "LINE_SEARCH_HORIZON_STEPS": "8",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1000",
            "LINE_SEARCH_LOG_LR_EMA": "1",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout8_smooth",
        "description": "8-step rollout with progress score and LR inertia",
        "env": {
            "LINE_SEARCH_INTERVAL": "16",
            "LINE_SEARCH_HORIZON_STEPS": "8",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0.05",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1.08",
            "LINE_SEARCH_LOG_LR_EMA": "0.35",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
    {
        "name": "rollout4_wide_grid",
        "description": "4-step rollout with wider LR-ratio grid",
        "env": {
            "LINE_SEARCH_INTERVAL": "8",
            "LINE_SEARCH_HORIZON_STEPS": "4",
            "LINE_SEARCH_HELD_OUT_BATCHES": "8",
            "LINE_SEARCH_VAL_BATCHES_PER_SEARCH": "4",
            "LINE_SEARCH_LR_RATIO_DEPTH1": "0.75",
            "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT": "0.05",
            "LINE_SEARCH_MAX_LOG_MOVE_MULT": "1.08",
            "LINE_SEARCH_LOG_LR_EMA": "0.35",
            "LINE_SEARCH_NORM_FEEDBACK": "0",
        },
    },
]


def run_line_search_sweep():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    failures = []
    print("Launching LR-search sweep:")
    for experiment in LINE_SEARCH_EXPERIMENTS:
        print(f"  - {experiment['name']}: {experiment['description']}")
    for experiment in LINE_SEARCH_EXPERIMENTS:
        name = experiment["name"]
        log_path = os.path.join(script_dir, f"baseline_line_search_{name}.log")
        env = os.environ.copy()
        env.update(experiment["env"])
        env["LINE_SEARCH_EXPERIMENT_NAME"] = name
        env["LINE_SEARCH_EXPERIMENT_DESCRIPTION"] = experiment["description"]
        print(f"\n=== starting {name}; log: {log_path} ===", flush=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(
                f"LINE_SEARCH_EXPERIMENT {json.dumps(experiment, sort_keys=True)}\n"
            )
            log_file.flush()
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=os.getcwd(),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"=== finished {name} with exit code {result.returncode} ===", flush=True)
        if result.returncode != 0:
            failures.append((name, result.returncode, log_path))
    if failures:
        print("\nSweep finished with failures:")
        for name, returncode, log_path in failures:
            print(f"  {name}: exit code {returncode}; log: {log_path}")
        sys.exit(1)
    print("\nSweep finished successfully.")


if "LINE_SEARCH_EXPERIMENT_NAME" not in os.environ:
    run_line_search_sweep()
    sys.exit(0)

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
                name="lm_head",
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=lm_head_wd,
            ),
            dict(
                name="embedding",
                kind="adamw",
                params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=embedding_wd,
            ),
            dict(
                name="value_embedding",
                kind="adamw",
                params=value_embeds_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=value_embedding_wd,
            ),
            dict(
                name="resid_lambdas",
                kind="adamw",
                params=resid_params,
                lr=scalar_lr * 0.01,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                name="x0_lambdas",
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
                    name=f"matrix_{tuple(shape)}",
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


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() not in {"0", "false", "no", "off"}


def env_str(name, default):
    return os.environ.get(name, default)


# Optimization
MAX_STEPS = env_int("MAX_STEPS", 1350)  # exact number of optimizer steps to train
TOTAL_BATCH_SIZE = 2**17  # ~524K tokens per optimizer step
LR_MULT = 1.5
EMBEDDING_LR = 0.6 * LR_MULT  # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004 * LR_MULT  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04 * LR_MULT  # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5 * LR_MULT  # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.1  # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95)  # Adam beta1, beta2
LM_HEAD_WD = 0.01  # AdamW weight decay for lm_head
EMBEDDING_WD = 0.001  # AdamW weight decay for token embeddings
VALUE_EMBEDDING_WD = 0.003  # AdamW weight decay for value embeddings

DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)
NORM_LOG_EVERY = 1  # optimizer steps between norm logs
INITIAL_LINE_SEARCH_LR_MULT = LR_MULT
LINE_SEARCH_EXPERIMENT_NAME = env_str("LINE_SEARCH_EXPERIMENT_NAME", "single")
LINE_SEARCH_EXPERIMENT_DESCRIPTION = env_str(
    "LINE_SEARCH_EXPERIMENT_DESCRIPTION",
    "single LR-search run",
)
LINE_SEARCH_DEPTH = env_int("LINE_SEARCH_DEPTH", 1)
LINE_SEARCH_LR_RATIOS_BY_DEPTH = {
    1: env_float("LINE_SEARCH_LR_RATIO_DEPTH1", 0.85),
    2: env_float("LINE_SEARCH_LR_RATIO_DEPTH2", 0.85),
}
LINE_SEARCH_LR_SUBGROUPS = ("all",)
LINE_SEARCH_VAL_SET_NAME = env_str("LINE_SEARCH_VAL_SET_NAME", "held_out_batch")
LINE_SEARCH_VAL_SET_DESCRIPTIONS = {
    "next_train_batch": "the next training batch",
    "held_out_batch": "rotating fixed held-out batches that are never trained on",
    "current_train_batch": "the current training batch used for gradients",
}
LINE_SEARCH_LOG_EVERY = 1  # optimizer steps between LR search logs; 0 disables logs

# Combined LR-search controller. It is intentionally scheduler-agnostic: the only
# "shape" prior is that LR should move smoothly unless the online rollout strongly
# prefers a change.
LINE_SEARCH_INTERVAL = env_int("LINE_SEARCH_INTERVAL", 8)
LINE_SEARCH_HORIZON_STEPS = env_int("LINE_SEARCH_HORIZON_STEPS", 4)
LINE_SEARCH_HELD_OUT_BATCHES = env_int("LINE_SEARCH_HELD_OUT_BATCHES", 8)
LINE_SEARCH_VAL_BATCHES_PER_SEARCH = env_int("LINE_SEARCH_VAL_BATCHES_PER_SEARCH", 4)
LINE_SEARCH_MAX_ROUNDS = env_int("LINE_SEARCH_MAX_ROUNDS", 2)
LINE_SEARCH_SAFE_LOSS_ATOL = env_float("LINE_SEARCH_SAFE_LOSS_ATOL", 3e-4)
LINE_SEARCH_SAFE_LOSS_RTOL = env_float("LINE_SEARCH_SAFE_LOSS_RTOL", 1e-4)
LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT = env_float(
    "LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT",
    0.05,
)
LINE_SEARCH_MIN_IMPROVEMENT = env_float("LINE_SEARCH_MIN_IMPROVEMENT", 1e-4)
LINE_SEARCH_MAX_LOG_MOVE = math.log(env_float("LINE_SEARCH_MAX_LOG_MOVE_MULT", 1.08))
LINE_SEARCH_LOG_LR_EMA = env_float("LINE_SEARCH_LOG_LR_EMA", 0.35)
LINE_SEARCH_NORM_FEEDBACK = env_bool("LINE_SEARCH_NORM_FEEDBACK", True)
LINE_SEARCH_NORM_WARMUP_STEPS = env_int("LINE_SEARCH_NORM_WARMUP_STEPS", 16)
LINE_SEARCH_NORM_TOLERANCE = env_float("LINE_SEARCH_NORM_TOLERANCE", 0.35)
LINE_SEARCH_NORM_MAX_LOG_MOVE = math.log(
    env_float("LINE_SEARCH_NORM_MAX_LOG_MOVE_MULT", 1.05)
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

optimizer = model.setup_optimizer(
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

uncompiled_model = model
model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
held_out_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")

print(f"Training steps: {MAX_STEPS}")
print(f"Gradient accumulation steps: {grad_accum_steps}")
print(f"Line-search experiment: {LINE_SEARCH_EXPERIMENT_NAME}")
print(f"Line-search experiment description: {LINE_SEARCH_EXPERIMENT_DESCRIPTION}")
print(f"Line-search depth: {LINE_SEARCH_DEPTH}")
print(f"Line-search subgroups: {LINE_SEARCH_LR_SUBGROUPS}")
print(f"Line-search interval: {LINE_SEARCH_INTERVAL}")
print(f"Line-search horizon steps: {LINE_SEARCH_HORIZON_STEPS}")
print(f"Line-search held-out batches: {LINE_SEARCH_HELD_OUT_BATCHES}")
print(f"Line-search val batches/search: {LINE_SEARCH_VAL_BATCHES_PER_SEARCH}")
print(f"Line-search norm feedback: {LINE_SEARCH_NORM_FEEDBACK}")
print(
    "Line-search validation set: "
    f"{LINE_SEARCH_VAL_SET_DESCRIPTIONS[LINE_SEARCH_VAL_SET_NAME]}"
)

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
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


def initial_line_search_lrs():
    return {
        subgroup: INITIAL_LINE_SEARCH_LR_MULT
        for subgroup in LINE_SEARCH_LR_SUBGROUPS
    }


def lr_tuple_to_dict(lr_tuple):
    return {
        subgroup: float(lr)
        for subgroup, lr in zip(LINE_SEARCH_LR_SUBGROUPS, lr_tuple)
    }


def lr_dict_to_tuple(lrs):
    return tuple(float(lrs[subgroup]) for subgroup in LINE_SEARCH_LR_SUBGROUPS)


def optimizer_lr_subgroup(group):
    return "all"


def update_optimizer_hyperparams(optimizer, step, subgroup_lrs):
    progress = min(step / max(1, MAX_STEPS - 1), 1.0)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)

    for group in optimizer.param_groups:
        subgroup = optimizer_lr_subgroup(group)
        lr_mult_base_lr = group["initial_lr"] / INITIAL_LINE_SEARCH_LR_MULT
        group["lr"] = lr_mult_base_lr * subgroup_lrs[subgroup]
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    return subgroup_lrs


def clone_optimizer_state_value(value):
    if torch.is_tensor(value):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {key: clone_optimizer_state_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clone_optimizer_state_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(clone_optimizer_state_value(val) for val in value)
    return deepcopy(value)


def snapshot_optimizer_state(optimizer):
    return {
        param: clone_optimizer_state_value(state)
        for param, state in optimizer.state.items()
    }


def restore_optimizer_state(optimizer, snapshot):
    optimizer.state.clear()
    for param, state in snapshot.items():
        optimizer.state[param] = clone_optimizer_state_value(state)


@torch.no_grad()
def snapshot_model_params(model):
    return [
        param.detach().clone(memory_format=torch.preserve_format)
        for param in model.parameters()
    ]


@torch.no_grad()
def restore_model_params(model, snapshot):
    for param, saved_param in zip(model.parameters(), snapshot):
        param.copy_(saved_param)


@torch.no_grad()
def snapshot_model_grads(model):
    return [
        None
        if param.grad is None
        else param.grad.detach().clone(memory_format=torch.preserve_format)
        for param in model.parameters()
    ]


@torch.no_grad()
def restore_model_grads(model, snapshot):
    for param, saved_grad in zip(model.parameters(), snapshot):
        if saved_grad is None:
            param.grad = None
        elif param.grad is None:
            param.grad = saved_grad.detach().clone(memory_format=torch.preserve_format)
        else:
            param.grad.copy_(saved_grad)


def collect_accumulated_batch(loader, grad_accum_steps):
    batch = []
    for _ in range(grad_accum_steps):
        x, y, epoch = next(loader)
        batch.append((x.detach().clone(), y.detach().clone(), epoch))
    return batch


def ensure_future_train_batches(
    future_train_batches, train_loader, grad_accum_steps, count
):
    while len(future_train_batches) < count:
        future_train_batches.append(
            collect_accumulated_batch(train_loader, grad_accum_steps)
        )


def flatten_accumulated_batches(accumulated_batches):
    return [
        micro_batch
        for accumulated_batch in accumulated_batches
        for micro_batch in accumulated_batch
    ]


@torch.no_grad()
def accumulated_batch_loss(model, batch, autocast_ctx):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    for x, y, _ in batch:
        with autocast_ctx:
            loss = model(x, y)
        total_loss += loss.detach().float().item()
    if was_training:
        model.train()
    return total_loss / len(batch)


def accumulate_batch_gradients(model, batch, autocast_ctx):
    was_training = model.training
    model.train()
    grad_accum_steps = len(batch)
    total_loss = 0.0
    for x, y, _ in batch:
        with autocast_ctx:
            loss = model(x, y)
        total_loss += loss.detach().float().item()
        (loss / grad_accum_steps).backward()
    if not was_training:
        model.eval()
    return total_loss / grad_accum_steps


def finite_loss(value):
    return math.isfinite(float(value))


def lrs_geomean(lrs):
    values = [
        max(float(lrs[subgroup]), 1e-12)
        for subgroup in LINE_SEARCH_LR_SUBGROUPS
    ]
    return math.exp(sum(math.log(value) for value in values) / len(values))


def add_trial_score(trial):
    val_loss = float(trial["val_loss"])
    if not finite_loss(val_loss):
        trial["score"] = float("inf")
        return trial
    train_loss = trial.get("rollout_train_loss")
    score = val_loss
    if train_loss is not None and finite_loss(train_loss):
        score += LINE_SEARCH_SCORE_TRAIN_LOSS_WEIGHT * float(train_loss)
    trial["score"] = score
    return trial


def set_line_search_lrs(optimizer, step, lr_tuple):
    update_optimizer_hyperparams(optimizer, step, lr_tuple_to_dict(lr_tuple))


def evaluate_trial_rollout(
    model,
    optimizer,
    step,
    rollout_train_batches,
    val_batch,
    autocast_ctx,
    baseline_params,
    baseline_optimizer_state,
    baseline_grads,
    lr_tuple,
):
    restore_model_params(model, baseline_params)
    restore_optimizer_state(optimizer, baseline_optimizer_state)
    restore_model_grads(model, baseline_grads)
    set_line_search_lrs(optimizer, step, lr_tuple)
    optimizer.step()
    train_losses = []
    horizon_steps = max(1, LINE_SEARCH_HORIZON_STEPS)
    for horizon_index in range(1, horizon_steps):
        model.zero_grad(set_to_none=True)
        train_loss = accumulate_batch_gradients(
            model,
            rollout_train_batches[horizon_index - 1],
            autocast_ctx,
        )
        train_losses.append(train_loss)
        set_line_search_lrs(optimizer, step + horizon_index, lr_tuple)
        optimizer.step()
    loss = accumulated_batch_loss(model, val_batch, autocast_ctx)
    restore_model_params(model, baseline_params)
    restore_optimizer_state(optimizer, baseline_optimizer_state)
    restore_model_grads(model, baseline_grads)
    rollout_train_loss = (
        sum(train_losses) / len(train_losses)
        if train_losses
        else None
    )
    return loss, rollout_train_loss


def trial_sort_key(trial):
    score = trial.get("score", trial["val_loss"])
    return score if math.isfinite(score) else float("inf")


def trial_lr_size_key(trial):
    return lrs_geomean(trial["lrs"])


def select_safe_trial(trials):
    finite_trials = [
        trial for trial in trials if finite_loss(trial["val_loss"])
    ]
    if not finite_trials:
        return min(trials, key=trial_sort_key)

    best_val_loss = min(float(trial["val_loss"]) for trial in finite_trials)
    safe_loss = (
        best_val_loss
        + LINE_SEARCH_SAFE_LOSS_ATOL
        + LINE_SEARCH_SAFE_LOSS_RTOL * abs(best_val_loss)
    )
    safe_trials = [
        trial
        for trial in finite_trials
        if float(trial["val_loss"]) <= safe_loss
    ]
    return max(safe_trials, key=trial_lr_size_key)


def line_search_lr_ratio(depth):
    if depth not in LINE_SEARCH_LR_RATIOS_BY_DEPTH:
        raise ValueError(f"No LR search ratio configured for depth={depth}.")
    return LINE_SEARCH_LR_RATIOS_BY_DEPTH[depth]


def offset_lr_tuple(center_lr_tuple, offset, lr_ratio):
    return tuple(
        lr * (lr_ratio ** exponent)
        for lr, exponent in zip(center_lr_tuple, offset)
    )


def lr_tuple_within_max(lr_tuple, max_lr_tuple):
    if max_lr_tuple is None:
        return True
    return all(
        lr <= max_lr * (1 + 1e-12)
        for lr, max_lr in zip(lr_tuple, max_lr_tuple)
    )


def line_search_neighbor_offsets(center_offset):
    for index in range(len(LINE_SEARCH_LR_SUBGROUPS)):
        down_offset = list(center_offset)
        down_offset[index] += 1
        yield tuple(down_offset)

        up_offset = list(center_offset)
        up_offset[index] -= 1
        yield tuple(up_offset)


def search_lrs_single_depth(
    model,
    optimizer,
    step,
    rollout_train_batches,
    val_batches,
    autocast_ctx,
    lr_center,
    lr_ratio,
    max_lr_tuple=None,
):
    val_batch = flatten_accumulated_batches(val_batches)
    baseline_params = snapshot_model_params(model)
    baseline_optimizer_state = snapshot_optimizer_state(optimizer)
    baseline_grads = snapshot_model_grads(model)
    center_lr_tuple = lr_dict_to_tuple(lr_center)
    center_offset = tuple(0 for _ in LINE_SEARCH_LR_SUBGROUPS)
    trial_cache = {}
    evaluated_trials = []
    search_rounds = []

    def evaluate_offset(offset):
        if offset in trial_cache:
            return trial_cache[offset]
        lr_tuple = offset_lr_tuple(center_lr_tuple, offset, lr_ratio)
        if not lr_tuple_within_max(lr_tuple, max_lr_tuple):
            return None
        loss, rollout_train_loss = evaluate_trial_rollout(
            model,
            optimizer,
            step,
            rollout_train_batches,
            val_batch,
            autocast_ctx,
            baseline_params,
            baseline_optimizer_state,
            baseline_grads,
            lr_tuple,
        )
        trial = {
            "offset": offset,
            "lrs": lr_tuple_to_dict(lr_tuple),
            "val_loss": loss,
            "rollout_train_loss": rollout_train_loss,
        }
        add_trial_score(trial)
        trial_cache[offset] = trial
        evaluated_trials.append(trial)
        return trial

    for _ in range(LINE_SEARCH_MAX_ROUNDS):
        middle_trial = evaluate_offset(center_offset)
        neighbor_trials = [
            trial
            for offset in line_search_neighbor_offsets(center_offset)
            for trial in [evaluate_offset(offset)]
            if trial is not None
        ]
        best_trial = select_safe_trial([middle_trial, *neighbor_trials])
        improvement = trial_sort_key(middle_trial) - trial_sort_key(best_trial)
        safe_larger_lr = (
            best_trial["offset"] != center_offset
            and trial_lr_size_key(best_trial) > trial_lr_size_key(middle_trial)
            and float(best_trial["val_loss"])
            <= float(middle_trial["val_loss"])
            + LINE_SEARCH_SAFE_LOSS_ATOL
            + LINE_SEARCH_SAFE_LOSS_RTOL * abs(float(middle_trial["val_loss"]))
        )
        improved = improvement > LINE_SEARCH_MIN_IMPROVEMENT or safe_larger_lr
        search_rounds.append(
            {
                "middle_offset": center_offset,
                "middle_lrs": middle_trial["lrs"],
                "middle_val_loss": middle_trial["val_loss"],
                "middle_score": middle_trial["score"],
                "best_offset": best_trial["offset"],
                "best_lrs": best_trial["lrs"],
                "best_val_loss": best_trial["val_loss"],
                "best_score": best_trial["score"],
                "improved": improved,
                "neighbor_offsets": [trial["offset"] for trial in neighbor_trials],
            }
        )
        if not improved:
            break
        center_offset = best_trial["offset"]

    restore_model_params(model, baseline_params)
    restore_optimizer_state(optimizer, baseline_optimizer_state)
    restore_model_grads(model, baseline_grads)
    best_lr_tuple = offset_lr_tuple(center_lr_tuple, center_offset, lr_ratio)
    best_lrs = lr_tuple_to_dict(best_lr_tuple)
    return {
        "lrs": best_lrs,
        "lr_tuple": best_lr_tuple,
        "val_loss": trial_cache[center_offset]["val_loss"],
        "score": trial_cache[center_offset]["score"],
        "rollout_train_loss": trial_cache[center_offset]["rollout_train_loss"],
        "trials": evaluated_trials,
        "search_rounds": search_rounds,
        "num_evaluated": len(evaluated_trials),
    }


def search_lrs_depth2(
    model,
    optimizer,
    step,
    second_train_batch,
    depth2_val_batches,
    autocast_ctx,
    lr_center,
):
    baseline_params = snapshot_model_params(model)
    baseline_optimizer_state = snapshot_optimizer_state(optimizer)
    baseline_grads = snapshot_model_grads(model)
    center_lr_tuple = lr_dict_to_tuple(lr_center)
    center_offset = tuple(0 for _ in LINE_SEARCH_LR_SUBGROUPS)
    depth1_lr_ratio = line_search_lr_ratio(1)
    depth2_lr_ratio = line_search_lr_ratio(2)
    trial_cache = {}
    evaluated_trials = []
    search_rounds = []

    def restore_baseline_node():
        restore_model_params(model, baseline_params)
        restore_optimizer_state(optimizer, baseline_optimizer_state)
        restore_model_grads(model, baseline_grads)

    def evaluate_offset(offset):
        if offset in trial_cache:
            return trial_cache[offset]

        lr_tuple = offset_lr_tuple(center_lr_tuple, offset, depth1_lr_ratio)
        try:
            restore_baseline_node()
            set_line_search_lrs(optimizer, step, lr_tuple)
            optimizer.step()
            model.zero_grad(set_to_none=True)
            accumulate_batch_gradients(model, second_train_batch, autocast_ctx)
            depth2_result = search_lrs_single_depth(
                model,
                optimizer,
                step + 1,
                [second_train_batch],
                depth2_val_batches,
                autocast_ctx,
                lr_tuple_to_dict(lr_tuple),
                depth2_lr_ratio,
                max_lr_tuple=lr_tuple,
            )
        finally:
            restore_baseline_node()

        trial = {
            "offset": offset,
            "lrs": lr_tuple_to_dict(lr_tuple),
            "val_loss": depth2_result["val_loss"],
            "depth2_result": depth2_result,
        }
        trial_cache[offset] = trial
        evaluated_trials.append(trial)
        return trial

    while True:
        middle_trial = evaluate_offset(center_offset)
        neighbor_trials = [
            evaluate_offset(offset)
            for offset in line_search_neighbor_offsets(center_offset)
        ]
        best_trial = min(
            [middle_trial, *neighbor_trials],
            key=trial_sort_key,
        )
        improved = trial_sort_key(best_trial) < trial_sort_key(middle_trial)
        search_rounds.append(
            {
                "middle_offset": center_offset,
                "middle_lrs": middle_trial["lrs"],
                "middle_depth2_best_lrs": middle_trial["depth2_result"]["lrs"],
                "middle_val_loss": middle_trial["val_loss"],
                "best_offset": best_trial["offset"],
                "best_lrs": best_trial["lrs"],
                "best_depth2_lrs": best_trial["depth2_result"]["lrs"],
                "best_val_loss": best_trial["val_loss"],
                "improved": improved,
                "neighbor_offsets": [trial["offset"] for trial in neighbor_trials],
            }
        )
        if not improved:
            break
        center_offset = best_trial["offset"]

    restore_baseline_node()
    best_trial = trial_cache[center_offset]
    return {
        "depth": 2,
        "lrs": best_trial["lrs"],
        "lr_tuple": lr_dict_to_tuple(best_trial["lrs"]),
        "val_loss": best_trial["val_loss"],
        "best_depth2_lrs": best_trial["depth2_result"]["lrs"],
        "best_depth2_lr_tuple": best_trial["depth2_result"]["lr_tuple"],
        "trials": evaluated_trials,
        "search_rounds": search_rounds,
        "num_evaluated": len(evaluated_trials),
        "depth2_num_evaluated": sum(
            trial["depth2_result"]["num_evaluated"]
            for trial in evaluated_trials
        ),
    }


def search_lrs(
    model,
    optimizer,
    step,
    rollout_train_batches,
    depth2_val_batches,
    autocast_ctx,
    lr_center,
):
    if LINE_SEARCH_DEPTH == 1:
        return search_lrs_single_depth(
            model,
            optimizer,
            step,
            rollout_train_batches,
            depth2_val_batches,
            autocast_ctx,
            lr_center,
            line_search_lr_ratio(1),
        )
    if LINE_SEARCH_DEPTH == 2:
        return search_lrs_depth2(
            model,
            optimizer,
            step,
            rollout_train_batches[0],
            depth2_val_batches,
            autocast_ctx,
            lr_center,
        )
    raise ValueError(f"Unsupported LINE_SEARCH_DEPTH={LINE_SEARCH_DEPTH}.")


def build_line_search_log(
    lr_search_result,
    step,
    val_set_name=None,
    num_batches=None,
    description=None,
):
    record = {
        "type": "lr_search",
        "experiment": LINE_SEARCH_EXPERIMENT_NAME,
        "step": step,
        "best_lrs": {
            subgroup: log_float(lr_search_result["lrs"][subgroup])
            for subgroup in LINE_SEARCH_LR_SUBGROUPS
        },
        "best_val_loss": log_float(lr_search_result["val_loss"]),
        "best_score": log_float(lr_search_result.get("score", lr_search_result["val_loss"])),
        "horizon_steps": LINE_SEARCH_HORIZON_STEPS,
        "search_interval": LINE_SEARCH_INTERVAL,
    }
    if "raw_lrs" in lr_search_result:
        record["raw_lrs"] = {
            subgroup: log_float(lr_search_result["raw_lrs"][subgroup])
            for subgroup in LINE_SEARCH_LR_SUBGROUPS
        }
    if "num_evaluated" in lr_search_result:
        record["num_evaluated"] = lr_search_result["num_evaluated"]
    if "depth" in lr_search_result:
        record["depth"] = lr_search_result["depth"]
    if "best_depth2_lrs" in lr_search_result:
        record["best_depth2_lrs"] = {
            subgroup: log_float(lr_search_result["best_depth2_lrs"][subgroup])
            for subgroup in LINE_SEARCH_LR_SUBGROUPS
        }
    if "depth2_num_evaluated" in lr_search_result:
        record["depth2_num_evaluated"] = lr_search_result["depth2_num_evaluated"]
    if val_set_name is not None:
        record["val_set"] = val_set_name
    if num_batches is not None:
        record["num_batches"] = num_batches
    if description is not None:
        record["description"] = description
    return record


def format_lr_summary(lr_by_subgroup):
    return ", ".join(
        f"{subgroup}={lr_by_subgroup[subgroup]:.5g}"
        for subgroup in LINE_SEARCH_LR_SUBGROUPS
    )


def required_future_train_batch_count(val_set_name):
    rollout_count = max(1, LINE_SEARCH_HORIZON_STEPS - 1)
    if val_set_name == "next_train_batch":
        return rollout_count + LINE_SEARCH_VAL_BATCHES_PER_SEARCH
    return rollout_count


def rotating_held_out_batches(held_out_batches, step):
    count = min(LINE_SEARCH_VAL_BATCHES_PER_SEARCH, len(held_out_batches))
    return [
        held_out_batches[(step + offset) % len(held_out_batches)]
        for offset in range(count)
    ]


def select_depth2_line_search_batches(
    val_set_name,
    train_batch,
    future_train_batches,
    held_out_batches,
    step,
):
    rollout_count = max(0, LINE_SEARCH_HORIZON_STEPS - 1)
    rollout_train_batches = future_train_batches[:rollout_count]
    second_train_batch = future_train_batches[0]
    if LINE_SEARCH_DEPTH == 1:
        if val_set_name == "next_train_batch":
            return rollout_train_batches, future_train_batches[
                rollout_count : rollout_count + LINE_SEARCH_VAL_BATCHES_PER_SEARCH
            ]
        if val_set_name == "held_out_batch":
            return rollout_train_batches, rotating_held_out_batches(
                held_out_batches,
                step,
            )
        if val_set_name == "current_train_batch":
            return rollout_train_batches, [train_batch]
        raise ValueError(f"Unknown line-search validation set: {val_set_name}")

    if val_set_name == "next_train_batch":
        return rollout_train_batches, future_train_batches[
            rollout_count : rollout_count + LINE_SEARCH_VAL_BATCHES_PER_SEARCH
        ]
    if val_set_name == "held_out_batch":
        return rollout_train_batches, rotating_held_out_batches(
            held_out_batches,
            step,
        )
    if val_set_name == "current_train_batch":
        return rollout_train_batches, [second_train_batch]
    raise ValueError(f"Unknown line-search validation set: {val_set_name}")


def smooth_line_search_lrs(previous_lrs, proposed_lrs):
    smoothed = {}
    for subgroup in LINE_SEARCH_LR_SUBGROUPS:
        previous = max(float(previous_lrs[subgroup]), 1e-12)
        proposed = max(float(proposed_lrs[subgroup]), 1e-12)
        prev_log = math.log(previous)
        proposed_log = math.log(proposed)
        delta = max(
            -LINE_SEARCH_MAX_LOG_MOVE,
            min(LINE_SEARCH_MAX_LOG_MOVE, proposed_log - prev_log),
        )
        target_log = prev_log + delta
        smoothed_log = (
            (1 - LINE_SEARCH_LOG_LR_EMA) * prev_log
            + LINE_SEARCH_LOG_LR_EMA * target_log
        )
        smoothed[subgroup] = math.exp(smoothed_log)
    return smoothed


def norm_log_subgroup(name):
    return "all"


def median(values):
    values = sorted(float(value) for value in values if finite_loss(value))
    if not values:
        return None
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return 0.5 * (values[midpoint - 1] + values[midpoint])


def aggregate_update_norms_by_subgroup(norm_log_record):
    grouped = {subgroup: [] for subgroup in LINE_SEARCH_LR_SUBGROUPS}
    if norm_log_record is None:
        return {subgroup: None for subgroup in LINE_SEARCH_LR_SUBGROUPS}
    for name, value in norm_log_record["update_norms"].items():
        subgroup = norm_log_subgroup(name)
        if subgroup in grouped and value is not None:
            grouped[subgroup].append(value)
    return {
        subgroup: median(values)
        for subgroup, values in grouped.items()
    }


def update_norm_feedback_reference(reference, update_norms, step):
    if reference is None:
        reference = {}
    for subgroup, value in update_norms.items():
        if value is None:
            continue
        if step < LINE_SEARCH_NORM_WARMUP_STEPS:
            reference[subgroup] = max(float(value), reference.get(subgroup, 0.0))
        elif subgroup not in reference:
            reference[subgroup] = float(value)
    return reference


def apply_norm_feedback(lrs, update_norms, reference):
    if not LINE_SEARCH_NORM_FEEDBACK or not reference:
        return lrs
    adjusted = dict(lrs)
    for subgroup in LINE_SEARCH_LR_SUBGROUPS:
        current = update_norms.get(subgroup)
        target = reference.get(subgroup)
        if current is None or target is None or current <= 0 or target <= 0:
            continue
        ratio = float(current) / float(target)
        if 1 - LINE_SEARCH_NORM_TOLERANCE <= ratio <= 1 + LINE_SEARCH_NORM_TOLERANCE:
            continue
        log_adjust = -0.5 * math.log(max(ratio, 1e-12))
        log_adjust = max(
            -LINE_SEARCH_NORM_MAX_LOG_MOVE,
            min(LINE_SEARCH_NORM_MAX_LOG_MOVE, log_adjust),
        )
        adjusted[subgroup] = max(adjusted[subgroup] * math.exp(log_adjust), 1e-12)
    return adjusted


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
train_batch = collect_accumulated_batch(train_loader, grad_accum_steps)
future_train_batches = []
held_out_batches = [
    collect_accumulated_batch(held_out_loader, grad_accum_steps)
    for _ in range(LINE_SEARCH_HELD_OUT_BATCHES)
]
lr_by_subgroup = initial_line_search_lrs()
norm_feedback_reference = None
last_lr_search_result = None
epoch = train_batch[-1][2]

while step < MAX_STEPS:
    torch.cuda.synchronize()
    t0 = time.time()
    should_log_norms = NORM_LOG_EVERY > 0 and step % NORM_LOG_EVERY == 0
    activation_norms = None
    residual_mix_l2_fractions = None
    residual_path_l2_fractions = None
    for micro_step, (x, y, epoch) in enumerate(train_batch):
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

    should_search_lrs = step == 0 or step % LINE_SEARCH_INTERVAL == 0
    required_future_count = (
        required_future_train_batch_count(LINE_SEARCH_VAL_SET_NAME)
        if should_search_lrs
        else 1
    )
    ensure_future_train_batches(
        future_train_batches,
        train_loader,
        grad_accum_steps,
        required_future_count,
    )
    lr_search_result = last_lr_search_result
    if should_search_lrs:
        rollout_train_batches, depth2_val_batches = select_depth2_line_search_batches(
            LINE_SEARCH_VAL_SET_NAME,
            train_batch,
            future_train_batches,
            held_out_batches,
            step,
        )
        lr_search_result = search_lrs(
            model,
            optimizer,
            step,
            rollout_train_batches,
            depth2_val_batches,
            autocast_ctx,
            lr_by_subgroup,
        )
        lr_search_result["raw_lrs"] = lr_search_result["lrs"]
        lr_search_result["lrs"] = smooth_line_search_lrs(
            lr_by_subgroup,
            lr_search_result["lrs"],
        )
        lr_by_subgroup = lr_search_result["lrs"]
        last_lr_search_result = lr_search_result
    update_optimizer_hyperparams(optimizer, step, lr_by_subgroup)
    if (
        should_search_lrs
        and LINE_SEARCH_LOG_EVERY > 0
        and step % LINE_SEARCH_LOG_EVERY == 0
    ):
        lr_log_record = build_line_search_log(
            lr_search_result,
            step,
            val_set_name=LINE_SEARCH_VAL_SET_NAME,
            num_batches=len(depth2_val_batches),
            description=LINE_SEARCH_VAL_SET_DESCRIPTIONS[LINE_SEARCH_VAL_SET_NAME],
        )
        print(f"\nLR_SEARCH {json.dumps(lr_log_record, sort_keys=True)}", flush=True)
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
        update_norms_by_subgroup = aggregate_update_norms_by_subgroup(norm_log_record)
        norm_feedback_reference = update_norm_feedback_reference(
            norm_feedback_reference,
            update_norms_by_subgroup,
            step,
        )
        if step >= LINE_SEARCH_NORM_WARMUP_STEPS:
            lr_by_subgroup = apply_norm_feedback(
                lr_by_subgroup,
                update_norms_by_subgroup,
                norm_feedback_reference,
            )
        print(f"\nNORM_LOG {json.dumps(norm_log_record, sort_keys=True)}", flush=True)
    model.zero_grad(set_to_none=True)
    train_batch = future_train_batches.pop(0)

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
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * min((step + 1) / MAX_STEPS, 1.0)
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining_steps = max(0, MAX_STEPS - step - 1)

    print(
        f"\rstep {step:05d} ({pct_done:.5g}%) | loss: {debiased_smooth_loss:.5g} | lrs: {format_lr_summary(lr_by_subgroup)} | val_loss: {lr_search_result['val_loss'] if lr_search_result is not None else float('nan'):.5g} | dt: {dt * 1000:.5g}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.5g}% | epoch: {epoch} | remaining_steps: {remaining_steps}    ",
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
print(f"training_seconds: {total_training_time:.5g}")
print(f"total_seconds:    {t_end - t_start:.5g}")
print(f"peak_vram_mb:     {peak_vram_mb:.5g}")
print(f"mfu_percent:      {steady_state_mfu:.5g}")
print(f"total_tokens_M:   {total_tokens / 1e6:.5g}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.5g}")
print(f"depth:            {DEPTH}")
print(f"experiment:       {LINE_SEARCH_EXPERIMENT_NAME}")
for subgroup in LINE_SEARCH_LR_SUBGROUPS:
    print(f"lr_{subgroup}:        {lr_by_subgroup[subgroup]:.5g}")
print(f"line_search_depth: {LINE_SEARCH_DEPTH}")
