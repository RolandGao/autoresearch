"""Learn a fixed-scale softmax with optimizer variants.

The model is a single bias-free linear layer. Inputs, target weights, and the
initial learned weights are row-normalized, but only the H optimizer variants
renormalize weights after each step. The softmax scalar is fixed to 10, so the
only trainable parameter is the linear weight.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


BASELINE_VARIANTS = ("AdamW", "SGD", "Muon")
ADAM_H_VARIANTS = ("AdamH_H1", "AdamH_H2")
MUON_H_VARIANTS = ("MuonH_H1", "MuonH_H2")
SGD_H_VARIANTS = tuple(f"SGDH_H{i}" for i in range(1, 11))
ALL_VARIANTS = BASELINE_VARIANTS + ADAM_H_VARIANTS + MUON_H_VARIANTS + SGD_H_VARIANTS
FULL_SEARCH_VARIANTS = ("AdamH_H1", "SGDH_H1")
FULL_SEARCH_NUM_SAMPLES = 30_000
FULL_SEARCH_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
FULL_SEARCH_RUNS_PER_BATCH_SIZE = 50
FULL_SEARCH_SGDH_MOMENTUMS = (0.0, 0.5, 0.7, 0.8, 0.9)
FULL_SEARCH_ADAM_BETA1S = (0.0, 0.5, 0.8, 0.9, 0.95)
FULL_SEARCH_ADAM_BETA2S = (0.9, 0.95, 0.99, 0.999)
FULL_SEARCH_ADAM_EPS = (1e-8, 1e-7, 1e-6)
FULL_SEARCH_SGDH_LR_SLOPE = 0.001409420528
FULL_SEARCH_SGDH_LR_INTERCEPT = 0.002388538963
FULL_SEARCH_ADAMH_LR_SLOPE = 0.001215494191
FULL_SEARCH_ADAMH_LR_INTERCEPT = 0.009600782619

BEST_HPARAMS: dict[str, dict[str, Any]] = {
    "AdamW": {
        "variant": "AdamW",
        "lr": 0.010240548631837793,
        "beta1": 0.0,
        "beta2": 0.95,
        "eps": 1e-7,
        "weight_decay": 0.0,
        "batch_size": 64,
        "steps": 913,
        "sample_mode": "shuffle_cycle",
    },
    "SGD": {
        "variant": "SGD",
        "lr": 7752.901539639679,
        "momentum": 0.95,
        "nesterov": False,
        "weight_decay": 0.0,
        "batch_size": 768,
        "steps": 480,
        "sample_mode": "shuffle_cycle",
    },
    "Muon": {
        "variant": "Muon",
        "lr": 0.034261406141903754,
        "momentum": 0.5,
        "nesterov": False,
        "ns_steps": 5,
        "weight_decay": 1.7629279980926629e-6,
        "batch_size": 64,
        "steps": 656,
        "lr_schedule": "exp_power",
        "lr_decay": 3.912023005428146,
        "lr_power": 1.0,
        "sample_mode": "shuffle_cycle",
    },
    "AdamH_H1": {
        "variant": "AdamH_H1",
        "lr": 0.008153784777378227,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-6,
        "weight_decay": 0.0,
        "batch_size": 16,
        "steps": 2750,
        "lr_schedule": "exp_power",
        "lr_decay": 3.912023005428146,
        "lr_power": 1.0,
        "sample_mode": "shuffle_cycle",
    },
    "AdamH_H2": {
        "variant": "AdamH_H2",
        "lr": 0.007139180639687829,
        "beta1": 0.95,
        "beta2": 0.99,
        "eps": 1e-6,
        "weight_decay": 0.0,
        "batch_size": 8,
        "steps": 3750,
        "lr_schedule": "exp_power",
        "lr_decay": 4.605170185988092,
        "lr_power": 1.0,
        "sample_mode": "shuffle_cycle",
    },
    "MuonH_H1": {
        "variant": "MuonH_H1",
        "lr": 0.03373468550935069,
        "momentum": 0.75,
        "nesterov": True,
        "ns_steps": 5,
        "weight_decay": 0.0,
        "batch_size": 32,
        "steps": 1125,
        "lr_schedule": "exp_power",
        "lr_decay": 4.605170185988092,
        "lr_power": 1.0,
        "sample_mode": "shuffle_cycle",
    },
    "MuonH_H2": {
        "variant": "MuonH_H2",
        "lr": 0.040759604048954214,
        "momentum": 0.0,
        "nesterov": True,
        "ns_steps": 3,
        "weight_decay": 0.0,
        "batch_size": 32,
        "steps": 1312,
        "lr_schedule": "exp_power",
        "lr_decay": 3.912023005428146,
        "lr_power": 1.0,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H1": {
        "variant": "SGDH_H1",
        "lr": 0.03199204297497002,
        "momentum": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
        "batch_size": 64,
        "steps": 543,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H2": {
        "variant": "SGDH_H2",
        "lr": 0.008893846680634183,
        "momentum": 0.8,
        "nesterov": True,
        "weight_decay": 0.0,
        "batch_size": 32,
        "steps": 1207,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H3": {
        "variant": "SGDH_H3",
        "lr": 0.03338232662336505,
        "momentum": 0.0,
        "nesterov": False,
        "g_projection": False,
        "g_norm": False,
        "weight_decay": 0.0,
        "batch_size": 64,
        "steps": 539,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H4": {
        "variant": "SGDH_H4",
        "lr": 0.03379977972858536,
        "momentum": 0.0,
        "nesterov": True,
        "g_projection": False,
        "g_norm": False,
        "weight_decay": 0.0,
        "batch_size": 64,
        "steps": 546,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H5": {
        "variant": "SGDH_H5",
        "lr": 0.0066160278093204475,
        "momentum": 0.8,
        "nesterov": False,
        "g_projection": False,
        "g_norm": True,
        "weight_decay": 0.0,
        "batch_size": 32,
        "steps": 1162,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H6": {
        "variant": "SGDH_H6",
        "lr": 0.02280895185750329,
        "momentum": 0.8,
        "nesterov": True,
        "g_projection": False,
        "g_norm": True,
        "weight_decay": 0.0,
        "batch_size": 128,
        "steps": 305,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H7": {
        "variant": "SGDH_H7",
        "lr": 0.03156033773294969,
        "momentum": 0.0,
        "nesterov": False,
        "g_projection": True,
        "g_norm": False,
        "weight_decay": 0.0,
        "batch_size": 64,
        "steps": 541,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H8": {
        "variant": "SGDH_H8",
        "lr": 0.026674570347838326,
        "momentum": 0.5,
        "nesterov": True,
        "g_projection": True,
        "g_norm": False,
        "weight_decay": 0.0,
        "batch_size": 64,
        "steps": 542,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H9": {
        "variant": "SGDH_H9",
        "lr": 0.00725358664493074,
        "momentum": 0.8,
        "nesterov": False,
        "g_projection": True,
        "g_norm": True,
        "weight_decay": 0.0,
        "batch_size": 32,
        "steps": 1126,
        "sample_mode": "shuffle_cycle",
    },
    "SGDH_H10": {
        "variant": "SGDH_H10",
        "lr": 0.006431237868811557,
        "momentum": 0.9,
        "nesterov": True,
        "g_projection": True,
        "g_norm": True,
        "weight_decay": 0.0,
        "batch_size": 32,
        "steps": 1187,
        "sample_mode": "shuffle_cycle",
    },
}


@dataclass
class Config:
    seed: int = 0
    input_dim: int = 128
    num_classes: int = 4000
    train_size: int = FULL_SEARCH_NUM_SAMPLES
    batch_size: int = 512
    steps: int = 2_000
    softmax_scale: float = 10.0
    ground_truth_softmax_scale: float = 10.0
    prob_noise_std: float = 1e-4
    eval_batch_size: int = 512
    runs_per_variant: int = 10
    search_rounds: int = 1
    optimizer_variants: tuple[str, ...] = FULL_SEARCH_VARIANTS
    hparam_mode: str = "full"
    step_mode: str = "config"
    shared_shape: bool = False
    target_rmse: float = 1.19e-5
    log_every: int = 0
    compile: bool = True
    device: str = "auto"


def parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--input-dim", type=int, default=Config.input_dim)
    parser.add_argument("--num-classes", type=int, default=Config.num_classes)
    parser.add_argument("--train-size", type=int, default=Config.train_size)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--softmax-scale", type=float, default=Config.softmax_scale)
    parser.add_argument(
        "--ground-truth-softmax-scale",
        type=float,
        default=Config.ground_truth_softmax_scale,
    )
    parser.add_argument("--prob-noise-std", type=float, default=Config.prob_noise_std)
    parser.add_argument("--eval-batch-size", type=int, default=Config.eval_batch_size)
    parser.add_argument("--runs-per-variant", type=int, default=Config.runs_per_variant)
    parser.add_argument("--search-rounds", type=int, default=Config.search_rounds)
    parser.add_argument(
        "--optimizer-variants",
        type=parse_str_tuple,
        default=Config.optimizer_variants,
        help="Comma-separated variants. Known: " + ",".join(ALL_VARIANTS),
    )
    parser.add_argument(
        "--hparam-mode",
        choices=("full", "random", "best"),
        default=Config.hparam_mode,
        help="Use the fixed full search, random search hparams, or saved best hparams.",
    )
    parser.add_argument(
        "--step-mode",
        choices=("config", "min"),
        default=Config.step_mode,
        help="Use --steps for every run or the saved per-variant minimum steps.",
    )
    parser.add_argument(
        "--shared-shape",
        action="store_true",
        help="Use --batch-size and --steps for every run, ignoring saved values.",
    )
    parser.add_argument("--target-rmse", type=float, default=Config.target_rmse)
    parser.add_argument("--log-every", type=int, default=Config.log_every)
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
        help="Disable torch.compile for the AdamH_H1/SGDH_H1 fast path.",
    )
    parser.add_argument("--device", type=str, default=Config.device)
    return Config(**vars(parser.parse_args()))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def row_normalize_update(update: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    update_norm = update.norm(dim=1, keepdim=True)
    normalized = update / update_norm.clamp_min(eps)
    return torch.where(update_norm > 0, normalized, torch.zeros_like(normalized))


def project_rows(update: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    denom = weight.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    return update - (update * weight).sum(dim=1, keepdim=True) * weight / denom


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_weight_decay(rng: random.Random, low: float, high: float) -> float:
    if rng.random() < 0.25:
        return 0.0
    return log_uniform(rng, low, high)


@torch.no_grad()
def build_problem(
    config: Config, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = normalize_rows(torch.randn(config.train_size, config.input_dim, device=device))
    ground_truth_weight = normalize_rows(
        torch.randn(config.num_classes, config.input_dim, device=device)
    )

    clean_target_probs = torch.empty(
        config.train_size, config.num_classes, device=device
    )
    noisy_target_probs = torch.empty(
        config.train_size, config.num_classes, device=device
    )
    for start in range(0, config.train_size, config.eval_batch_size):
        end = min(start + config.eval_batch_size, config.train_size)
        clean_logits = config.ground_truth_softmax_scale * F.linear(
            x[start:end], ground_truth_weight
        )
        clean_probs = clean_logits.softmax(dim=1)
        clean_target_probs[start:end] = clean_probs
        noisy_target_probs[start:end] = make_noisy_probs(
            clean_probs, config.prob_noise_std
        )

    return x, ground_truth_weight, clean_target_probs, noisy_target_probs


def make_noisy_probs(clean_probs: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std == 0:
        return clean_probs
    noisy_probs = clean_probs + noise_std * torch.randn_like(clean_probs)
    noisy_probs = noisy_probs.clamp_min(0)
    row_sums = noisy_probs.sum(dim=1, keepdim=True)

    empty_rows = row_sums.squeeze(1) <= 0
    if empty_rows.any():
        noisy_probs[empty_rows] = clean_probs[empty_rows]
        row_sums = noisy_probs.sum(dim=1, keepdim=True)

    return noisy_probs / row_sums.clamp_min(1e-12)


def probability_rmse_loss(
    logits: torch.Tensor, target_probs: torch.Tensor
) -> torch.Tensor:
    output_probs = logits.softmax(dim=1)
    return (output_probs - target_probs).square().mean().sqrt()


@torch.compile(dynamic=True, fullgraph=True)
def probability_rmse_loss_compiled(
    weight: torch.Tensor,
    x_batch: torch.Tensor,
    target_probs: torch.Tensor,
    softmax_scale: torch.Tensor,
) -> torch.Tensor:
    logits = softmax_scale * F.linear(x_batch, weight).float()
    output_probs = logits.softmax(dim=1)
    return (output_probs - target_probs).square().mean().sqrt()


@torch.compile(dynamic=True, fullgraph=True)
def softmax_probs_compiled(
    weight: torch.Tensor,
    x_batch: torch.Tensor,
    softmax_scale: torch.Tensor,
) -> torch.Tensor:
    logits = softmax_scale * F.linear(x_batch, weight).float()
    return logits.softmax(dim=1)


@torch.compile(dynamic=False, fullgraph=True)
def adam_h1_step_compiled(
    weight: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    lr_t: torch.Tensor,
    beta1_t: torch.Tensor,
    beta2_t: torch.Tensor,
    eps_t: torch.Tensor,
    lr_factor_t: torch.Tensor,
) -> None:
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps_t)
    update_norm = update.norm(dim=1, keepdim=True)
    normalized_update = update / update_norm.clamp_min(1e-12)
    normalized_update = torch.where(
        update_norm > 0, normalized_update, torch.zeros_like(normalized_update)
    )
    weight.add_(normalized_update * (-lr_t * lr_factor_t))
    weight.copy_(weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-12))


@torch.compile(dynamic=False, fullgraph=True)
def sgdh_h1_step_compiled(
    weight: torch.Tensor,
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    lr_t: torch.Tensor,
    momentum_t: torch.Tensor,
    lr_factor_t: torch.Tensor,
) -> None:
    momentum_buffer.mul_(momentum_t).add_(grad)
    update = momentum_buffer
    update_norm = update.norm(dim=1, keepdim=True)
    normalized_update = update / update_norm.clamp_min(1e-12)
    normalized_update = torch.where(
        update_norm > 0, normalized_update, torch.zeros_like(normalized_update)
    )
    weight.add_(normalized_update * (-lr_t * lr_factor_t))
    weight.copy_(weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-12))


class FixedScaleSoftmax(nn.Module):
    def __init__(self, config: Config, initial_weight: torch.Tensor):
        super().__init__()
        expected_shape = (config.num_classes, config.input_dim)
        if tuple(initial_weight.shape) != expected_shape:
            raise ValueError(
                f"initial_weight has shape {tuple(initial_weight.shape)}, "
                f"expected {expected_shape}"
            )
        self.softmax_scale = float(config.softmax_scale)
        self.weight = nn.Parameter(initial_weight.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_scale * F.linear(x, self.weight).float()

    @torch.no_grad()
    def normalize_weight_(self) -> None:
        self.weight.copy_(normalize_rows(self.weight))


polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def zeropower_via_newtonschulz5(g: torch.Tensor, ns_steps: int) -> torch.Tensor:
    x = g.float()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if x.size(-2) > x.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.to(dtype=g.dtype)


class BaselineMuon:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        momentum: float,
        weight_decay: float,
        ns_steps: int,
    ):
        self.param = param
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.momentum_buffer = torch.zeros_like(param)

    def zero_grad(self) -> None:
        self.param.grad = None

    @torch.no_grad()
    def step(self, lr_factor: float) -> None:
        if self.param.grad is None:
            return
        self.momentum_buffer.lerp_(self.param.grad, 1 - self.momentum)
        update = self.param.grad.lerp(self.momentum_buffer, self.momentum)
        update = zeropower_via_newtonschulz5(update, self.ns_steps)
        lr = (
            self.lr
            * lr_factor
            * max(1.0, self.param.size(0) / self.param.size(1)) ** 0.5
        )
        if self.weight_decay:
            self.param.mul_(1 - lr * self.weight_decay)
        self.param.add_(update, alpha=-lr)


class AdamH:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        h_variant: str,
    ):
        self.param = param
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.h_variant = h_variant
        self.step_count = 0
        self.exp_avg = torch.zeros_like(param)
        self.exp_avg_sq = torch.zeros_like(param)

    def zero_grad(self) -> None:
        self.param.grad = None

    @torch.no_grad()
    def step(self, lr_factor: float) -> None:
        if self.param.grad is None:
            return
        grad = self.param.grad
        self.step_count += 1
        self.exp_avg.lerp_(grad, 1 - self.beta1)
        self.exp_avg_sq.lerp_(grad.square(), 1 - self.beta2)
        bias1 = 1 - self.beta1**self.step_count
        bias2 = 1 - self.beta2**self.step_count
        update = (self.exp_avg / bias1) / ((self.exp_avg_sq / bias2).sqrt() + self.eps)
        if self.h_variant == "H2":
            update = project_rows(update, self.param)
        self.param.add_(row_normalize_update(update), alpha=-self.lr * lr_factor)
        self.param.copy_(normalize_rows(self.param))


class MuonH:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        momentum: float,
        ns_steps: int,
        h_variant: str,
    ):
        self.param = param
        self.lr = lr
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.h_variant = h_variant
        self.momentum_buffer = torch.zeros_like(param)

    def zero_grad(self) -> None:
        self.param.grad = None

    @torch.no_grad()
    def step(self, lr_factor: float) -> None:
        if self.param.grad is None:
            return
        self.momentum_buffer.lerp_(self.param.grad, 1 - self.momentum)
        update = self.param.grad.lerp(self.momentum_buffer, self.momentum)
        update = zeropower_via_newtonschulz5(update, self.ns_steps)
        if self.h_variant == "H2":
            update = project_rows(update, self.param)
        self.param.add_(row_normalize_update(update), alpha=-self.lr * lr_factor)
        self.param.copy_(normalize_rows(self.param))


class SGDH:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        momentum: float,
        h_index: int,
        nesterov: bool,
    ):
        self.param = param
        self.lr = lr
        self.momentum = momentum
        self.h_index = h_index
        self.nesterov = nesterov
        self.momentum_buffer = torch.zeros_like(param)
        if h_index >= 3:
            bit_index = h_index - 3
            self.g_projection = bool(bit_index & 4)
            self.g_norm = bool(bit_index & 2)
            self.nesterov = bool(bit_index & 1)
        else:
            self.g_projection = False
            self.g_norm = False

    def zero_grad(self) -> None:
        self.param.grad = None

    @torch.no_grad()
    def step(self, lr_factor: float) -> None:
        if self.param.grad is None:
            return
        grad = self.param.grad
        if self.h_index == 1:
            self.momentum_buffer.mul_(self.momentum).add_(grad)
            update = (
                grad + self.momentum * self.momentum_buffer
                if self.nesterov
                else self.momentum_buffer
            )
        elif self.h_index == 2:
            self.momentum_buffer.mul_(self.momentum).add_(grad)
            update = (
                grad + self.momentum * self.momentum_buffer
                if self.nesterov
                else self.momentum_buffer
            )
            update = project_rows(update, self.param)
        else:
            grad = project_rows(grad, self.param) if self.g_projection else grad
            grad = row_normalize_update(grad) if self.g_norm else grad
            self.momentum_buffer.mul_(self.momentum).add_(grad)
            self.momentum_buffer.copy_(project_rows(self.momentum_buffer, self.param))
            update = (
                grad + self.momentum * self.momentum_buffer
                if self.nesterov
                else self.momentum_buffer
            )

        self.param.add_(row_normalize_update(update), alpha=-self.lr * lr_factor)
        self.param.copy_(normalize_rows(self.param))


def make_optimizer(
    model: FixedScaleSoftmax, hparams: dict[str, Any]
) -> torch.optim.Optimizer | BaselineMuon | AdamH | MuonH | SGDH:
    variant = hparams["variant"]
    lr = float(hparams["lr"])
    if variant == "AdamW":
        return torch.optim.AdamW(
            [model.weight],
            lr=lr,
            betas=(float(hparams["beta1"]), float(hparams["beta2"])),
            eps=float(hparams["eps"]),
            weight_decay=float(hparams["weight_decay"]),
        )
    if variant == "SGD":
        momentum = float(hparams["momentum"])
        return torch.optim.SGD(
            [model.weight],
            lr=lr,
            momentum=momentum,
            weight_decay=float(hparams["weight_decay"]),
            nesterov=bool(hparams["nesterov"]) and momentum > 0,
        )
    if variant == "Muon":
        return BaselineMuon(
            model.weight,
            lr=lr,
            momentum=float(hparams["momentum"]),
            weight_decay=float(hparams["weight_decay"]),
            ns_steps=int(hparams["ns_steps"]),
        )
    if variant.startswith("AdamH_"):
        return AdamH(
            model.weight,
            lr=lr,
            betas=(float(hparams["beta1"]), float(hparams["beta2"])),
            eps=float(hparams["eps"]),
            h_variant=str(variant).split("_", 1)[1],
        )
    if variant.startswith("MuonH_"):
        return MuonH(
            model.weight,
            lr=lr,
            momentum=float(hparams["momentum"]),
            ns_steps=int(hparams["ns_steps"]),
            h_variant=str(variant).split("_", 1)[1],
        )
    if variant.startswith("SGDH_H"):
        return SGDH(
            model.weight,
            lr=lr,
            momentum=float(hparams["momentum"]),
            h_index=int(str(variant).rsplit("H", 1)[1]),
            nesterov=bool(hparams["nesterov"]),
        )
    raise ValueError(f"unknown optimizer variant: {variant}")


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    if isinstance(optimizer, torch.optim.Optimizer):
        for group in optimizer.param_groups:
            group["lr"] = lr


def linear_decay_factor(step: int, steps: int) -> float:
    return 1.0 - step / max(1, steps)


def lr_schedule_factor(step: int, steps: int, hparams: dict[str, Any]) -> float:
    schedule = str(hparams.get("lr_schedule", "linear"))
    progress = step / max(1, steps)
    if schedule == "linear":
        return linear_decay_factor(step, steps)
    if schedule == "constant":
        return 1.0
    if schedule == "exp_power":
        decay = float(hparams.get("lr_decay", 4.0))
        power = float(hparams.get("lr_power", 1.0))
        return math.exp(-decay * progress**power)
    raise ValueError(f"unknown lr_schedule: {schedule}")


def next_batch_indices(
    x_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
    sample_state: dict[str, Any],
    sample_mode: str,
) -> torch.Tensor:
    if sample_mode == "randint":
        return torch.randint(
            x_size,
            (batch_size,),
            device=device,
            generator=generator,
        )
    if sample_mode in ("shuffle_cycle", "fixed_cycle"):
        parts = []
        remaining = batch_size
        while remaining > 0:
            cursor = int(sample_state.get("cursor", 0))
            permutation = sample_state.get("permutation")
            if permutation is None:
                permutation = torch.randperm(x_size, device=device, generator=generator)
                cursor = 0
                sample_state["permutation"] = permutation
            elif cursor >= x_size:
                if sample_mode == "shuffle_cycle":
                    permutation = torch.randperm(
                        x_size, device=device, generator=generator
                    )
                    sample_state["permutation"] = permutation
                cursor = 0

            take = min(remaining, x_size - cursor)
            parts.append(permutation[cursor : cursor + take])
            cursor += take
            remaining -= take
            sample_state["cursor"] = cursor

        return torch.cat(parts) if len(parts) > 1 else parts[0]
    raise ValueError(f"unknown sample_mode: {sample_mode}")


@torch.no_grad()
def evaluate_clean_rmse(
    model: FixedScaleSoftmax,
    x: torch.Tensor,
    clean_target_probs: torch.Tensor,
    config: Config,
) -> float:
    model.eval()
    squared_error_sum = 0.0
    num_values = 0

    for start in range(0, x.size(0), config.eval_batch_size):
        end = min(start + config.eval_batch_size, x.size(0))
        output_probs = model(x[start:end]).softmax(dim=1)
        diff = output_probs - clean_target_probs[start:end]
        squared_error_sum += diff.square().sum().item()
        num_values += diff.numel()

    model.train()
    return math.sqrt(squared_error_sum / num_values)


@torch.no_grad()
def evaluate_clean_rmse_compiled(
    weight: torch.Tensor,
    x: torch.Tensor,
    clean_target_probs: torch.Tensor,
    config: Config,
) -> float:
    squared_error_sum = 0.0
    num_values = 0
    softmax_scale_t = torch.tensor(
        config.softmax_scale, dtype=torch.float32, device="cpu"
    )

    for start in range(0, x.size(0), config.eval_batch_size):
        end = min(start + config.eval_batch_size, x.size(0))
        output_probs = softmax_probs_compiled(weight, x[start:end], softmax_scale_t)
        diff = output_probs - clean_target_probs[start:end]
        squared_error_sum += diff.square().sum().item()
        num_values += diff.numel()

    return math.sqrt(squared_error_sum / num_values)


def base_space(variant: str) -> dict[str, tuple[float, float]]:
    if variant == "AdamW":
        return {"lr": (1e-4, 5e-2), "weight_decay": (1e-7, 1e-1)}
    if variant == "SGD":
        return {"lr": (1e-3, 1e5), "weight_decay": (1e-8, 1e-1)}
    if variant == "Muon":
        return {"lr": (1e-4, 2e-1), "weight_decay": (1e-8, 1e-1)}
    if variant.startswith("AdamH_"):
        return {"lr": (1e-5, 1e-1)}
    if variant.startswith("MuonH_"):
        return {"lr": (1e-5, 1e-1)}
    if variant.startswith("SGDH_"):
        return {"lr": (1e-5, 1.0)}
    raise ValueError(f"unknown variant: {variant}")


def sample_hparams(
    variant: str,
    rng: random.Random,
    previous_best: dict[str, Any] | None,
    round_idx: int,
) -> dict[str, Any]:
    space = base_space(variant)
    if previous_best is not None and round_idx > 0:
        lr = float(previous_best["lr"])
        lr_low, lr_high = space["lr"]
        tuned_low = max(lr_low, lr / 4)
        tuned_high = min(lr_high, lr * 4)
        if tuned_low >= tuned_high:
            tuned_low, tuned_high = lr_low, lr_high
        space = {**space, "lr": (tuned_low, tuned_high)}
        if "weight_decay" in space:
            wd = max(float(previous_best["weight_decay"]), 1e-12)
            wd_low, wd_high = base_space(variant)["weight_decay"]
            space["weight_decay"] = (max(wd_low, wd / 10), min(wd_high, wd * 10))

    hparams: dict[str, Any] = {
        "variant": variant,
        "lr": log_uniform(rng, *space["lr"]),
    }
    if "weight_decay" in space:
        hparams["weight_decay"] = sample_weight_decay(rng, *space["weight_decay"])
    else:
        hparams["weight_decay"] = 0.0

    if variant.startswith("Adam"):
        hparams["beta1"] = rng.choice((0.0, 0.5, 0.8, 0.9, 0.95))
        hparams["beta2"] = rng.choice((0.9, 0.95, 0.99, 0.999))
        hparams["eps"] = rng.choice((1e-8, 1e-7, 1e-6))
    else:
        momentum_choices = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99)
        if previous_best is not None and round_idx > 0:
            best_momentum = float(previous_best.get("momentum", 0.9))
            nearby = [
                best_momentum,
                max(0.0, best_momentum - 0.05),
                min(0.99, best_momentum + 0.05),
            ]
            hparams["momentum"] = rng.choice(tuple(momentum_choices) + tuple(nearby))
        else:
            hparams["momentum"] = rng.choice(momentum_choices)
        hparams["nesterov"] = rng.choice((False, True))
        hparams["ns_steps"] = rng.choice((3, 4, 5))

    if variant.startswith("SGDH_H") and int(variant.rsplit("H", 1)[1]) >= 3:
        h_index = int(variant.rsplit("H", 1)[1])
        bit_index = h_index - 3
        hparams["g_projection"] = bool(bit_index & 4)
        hparams["g_norm"] = bool(bit_index & 2)
        hparams["nesterov"] = bool(bit_index & 1)

    hparams["lr_schedule"] = rng.choice(("linear", "exp_power"))
    if hparams["lr_schedule"] == "exp_power":
        hparams["lr_decay"] = rng.uniform(0.5, 8.0)
        hparams["lr_power"] = rng.uniform(0.4, 2.5)
    hparams["sample_mode"] = rng.choice(("randint", "shuffle_cycle", "fixed_cycle"))

    return hparams


def rounded_steps_for_num_samples(num_samples: int, batch_size: int) -> int:
    return max(1, int(round(num_samples / batch_size)))


def full_search_base_hparams(variant: str) -> dict[str, Any]:
    hparams: dict[str, Any] = {
        "variant": variant,
        "weight_decay": 0.0,
        "lr_schedule": "exp_power",
        "sample_mode": "fixed_cycle",
    }
    if variant == "AdamH_H1":
        return hparams
    if variant == "SGDH_H1":
        hparams.update(
            {
                "nesterov": False,
            }
        )
        return hparams
    raise ValueError(
        "full hparam search supports only: " + ", ".join(FULL_SEARCH_VARIANTS)
    )


def predicted_full_search_lr(variant: str, batch_size: int) -> float:
    if variant == "AdamH_H1":
        return FULL_SEARCH_ADAMH_LR_SLOPE * batch_size + FULL_SEARCH_ADAMH_LR_INTERCEPT
    if variant == "SGDH_H1":
        return FULL_SEARCH_SGDH_LR_SLOPE * batch_size + FULL_SEARCH_SGDH_LR_INTERCEPT
    raise ValueError(
        "full hparam search supports only: " + ", ".join(FULL_SEARCH_VARIANTS)
    )


def full_search_samples_for_batch(
    config: Config, variant: str, batch_size: int
) -> list[dict[str, Any]]:
    variant_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(variant))
    rng = random.Random(config.seed + 1_000_003 + variant_seed + 1009 * batch_size)
    predicted_lr = predicted_full_search_lr(variant, batch_size)
    samples = []
    for sample_idx in range(FULL_SEARCH_RUNS_PER_BATCH_SIZE):
        hparams: dict[str, Any] = {
            "sample_idx": sample_idx,
            "continuous_sample_idx": sample_idx,
            "predicted_lr": predicted_lr,
            "lr": log_uniform(rng, predicted_lr / 10, predicted_lr * 10),
            "lr_decay": rng.uniform(2.5, 6.0),
            "lr_power": rng.uniform(0.8, 1.2),
        }
        if variant == "AdamH_H1":
            hparams.update(
                {
                    "beta1": rng.choice(FULL_SEARCH_ADAM_BETA1S),
                    "beta2": rng.choice(FULL_SEARCH_ADAM_BETA2S),
                    "eps": rng.choice(FULL_SEARCH_ADAM_EPS),
                }
            )
        elif variant == "SGDH_H1":
            hparams["momentum"] = rng.choice(FULL_SEARCH_SGDH_MOMENTUMS)
        else:
            raise ValueError(
                "full hparam search supports only: " + ", ".join(FULL_SEARCH_VARIANTS)
            )
        samples.append(hparams)
    return samples


def build_full_search_hparams(config: Config) -> list[dict[str, Any]]:
    candidate_specs: list[dict[str, Any]] = []
    for variant in config.optimizer_variants:
        for batch_size in FULL_SEARCH_BATCH_SIZES:
            steps = rounded_steps_for_num_samples(FULL_SEARCH_NUM_SAMPLES, batch_size)
            samples = full_search_samples_for_batch(config, variant, batch_size)
            for sample in samples:
                hparams = full_search_base_hparams(variant)
                hparams.update(sample)
                hparams.update(
                    {
                        "round": 1,
                        "batch_size": batch_size,
                        "steps": steps,
                        "num_samples": FULL_SEARCH_NUM_SAMPLES,
                    }
                )
                candidate_specs.append(hparams)
    return candidate_specs


def validate_variants(variants: tuple[str, ...]) -> None:
    unknown = sorted(set(variants) - set(ALL_VARIANTS))
    if unknown:
        raise ValueError(
            "unknown optimizer variants: "
            + ", ".join(unknown)
            + "; known variants: "
            + ", ".join(ALL_VARIANTS)
        )


def saved_hparams(variant: str, sample_idx: int) -> dict[str, Any]:
    hparams = dict(BEST_HPARAMS[variant])
    hparams["sample_idx"] = sample_idx
    return hparams


def attach_min_steps(hparams: dict[str, Any]) -> dict[str, Any]:
    hparams = dict(hparams)
    hparams["steps"] = int(hparams["steps"])
    return hparams


def supports_compiled_run(hparams: dict[str, Any]) -> bool:
    variant = str(hparams["variant"])
    if variant == "AdamH_H1":
        return True
    if variant == "SGDH_H1":
        return not bool(hparams.get("nesterov", False))
    return False


def train_one_run_compiled(
    hparams: dict[str, Any],
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    clean_target_probs: torch.Tensor,
    initial_weight: torch.Tensor,
    config: Config,
    device: torch.device,
    candidate_idx: int,
    num_candidates: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    weight = initial_weight.detach().clone().to(device).requires_grad_(True)
    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(config.seed + 10_000)

    scalar_device = torch.device("cpu")
    softmax_scale_t = torch.tensor(
        config.softmax_scale, dtype=torch.float32, device=scalar_device
    )
    lr_t = torch.tensor(float(hparams["lr"]), dtype=torch.float32, device=scalar_device)
    lr_factor_t = torch.tensor(1.0, dtype=torch.float32, device=scalar_device)

    variant = str(hparams["variant"])
    if variant == "AdamH_H1":
        exp_avg = torch.zeros_like(weight)
        exp_avg_sq = torch.zeros_like(weight)
        step_t = torch.tensor(0.0, dtype=torch.float32, device=scalar_device)
        beta1_t = torch.tensor(
            float(hparams["beta1"]), dtype=torch.float32, device=scalar_device
        )
        beta2_t = torch.tensor(
            float(hparams["beta2"]), dtype=torch.float32, device=scalar_device
        )
        eps_t = torch.tensor(
            float(hparams["eps"]), dtype=torch.float32, device=scalar_device
        )
    elif variant == "SGDH_H1":
        momentum_buffer = torch.zeros_like(weight)
        momentum_t = torch.tensor(
            float(hparams["momentum"]), dtype=torch.float32, device=scalar_device
        )
    else:
        raise ValueError(f"compiled training does not support variant: {variant}")

    last_loss = float("nan")
    base_lr = float(hparams["lr"])
    steps = int(hparams.get("steps", config.steps))
    batch_size = int(hparams.get("batch_size", config.batch_size))
    sample_mode = str(hparams.get("sample_mode", "randint"))
    sample_state: dict[str, Any] = {}
    training_start_time = time.time()
    for step in range(steps):
        lr_factor = lr_schedule_factor(step, steps, hparams)
        lr_factor_t.fill_(lr_factor)
        batch_idx = next_batch_indices(
            x.size(0),
            batch_size,
            device=device,
            generator=batch_generator,
            sample_state=sample_state,
            sample_mode=sample_mode,
        )
        weight.grad = None
        loss = probability_rmse_loss_compiled(
            weight, x[batch_idx], noisy_target_probs[batch_idx], softmax_scale_t
        )
        loss.backward()
        grad = weight.grad
        if grad is None:
            raise RuntimeError("compiled loss did not produce a weight gradient")

        with torch.no_grad():
            if variant == "AdamH_H1":
                step_t.fill_(step + 1)
                adam_h1_step_compiled(
                    weight,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    step_t,
                    lr_t,
                    beta1_t,
                    beta2_t,
                    eps_t,
                    lr_factor_t,
                )
            else:
                sgdh_h1_step_compiled(
                    weight,
                    grad,
                    momentum_buffer,
                    lr_t,
                    momentum_t,
                    lr_factor_t,
                )

        last_loss = float(loss.detach().item())
        should_log = config.log_every > 0 and (
            (step + 1) % config.log_every == 0 or step + 1 == steps
        )
        if should_log:
            print(
                f"candidate {candidate_idx + 1:04d}/{num_candidates:04d} "
                f"{hparams['variant']} step {step + 1:05d}/{steps:05d} "
                f"loss_rmse={last_loss:.8g} lr={base_lr * lr_factor:.6g}",
                flush=True,
            )

    weight.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed_sec = time.time() - training_start_time
    clean_rmse = evaluate_clean_rmse_compiled(
        weight.detach(), x, clean_target_probs, config
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - start_time
    peak_allocated_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved_bytes = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    weight_norms = weight.detach().norm(dim=1)
    summary = {
        **hparams,
        "candidate_idx": candidate_idx,
        "steps": steps,
        "batch_size": batch_size,
        "num_samples": int(hparams.get("num_samples", steps * batch_size)),
        "actual_samples": steps * batch_size,
        "compiled": True,
        "final_train_loss": last_loss,
        "clean_rmse": clean_rmse,
        "clean_train_rmse": clean_rmse,
        "target_met": clean_rmse <= config.target_rmse,
        "weight_row_norm_mean": float(weight_norms.mean().item()),
        "weight_row_norm_std": float(weight_norms.std().item()),
        "duration_sec": elapsed_sec,
        "training_elapsed_sec": training_elapsed_sec,
        "elapsed_sec": elapsed_sec,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_mib": peak_allocated_bytes / 1024**2,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_reserved_mib": peak_reserved_bytes / 1024**2,
    }
    print(f"RUN_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


def warmup_compiled_training(
    config: Config,
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    initial_weight: torch.Tensor,
    device: torch.device,
) -> None:
    if not config.compile or config.hparam_mode != "full":
        return

    variants = tuple(
        variant
        for variant in config.optimizer_variants
        if supports_compiled_run(full_search_base_hparams(variant))
    )
    if not variants:
        return

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.time()
    print(
        "COMPILE_WARMUP_START "
        + json.dumps(
            {
                "batch_sizes": FULL_SEARCH_BATCH_SIZES,
                "optimizer_variants": variants,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    scalar_device = torch.device("cpu")
    softmax_scale_t = torch.tensor(
        config.softmax_scale, dtype=torch.float32, device=scalar_device
    )
    lr_t = torch.tensor(0.01, dtype=torch.float32, device=scalar_device)
    lr_factor_t = torch.tensor(1.0, dtype=torch.float32, device=scalar_device)
    step_t = torch.tensor(1.0, dtype=torch.float32, device=scalar_device)
    beta1_t = torch.tensor(0.9, dtype=torch.float32, device=scalar_device)
    beta2_t = torch.tensor(0.999, dtype=torch.float32, device=scalar_device)
    eps_t = torch.tensor(1e-6, dtype=torch.float32, device=scalar_device)
    momentum_t = torch.tensor(0.0, dtype=torch.float32, device=scalar_device)

    for batch_size in FULL_SEARCH_BATCH_SIZES:
        batch_idx = torch.arange(batch_size, device=device) % x.size(0)
        x_batch = x[batch_idx]
        target_batch = noisy_target_probs[batch_idx]
        for variant in variants:
            weight = initial_weight.detach().clone().to(device).requires_grad_(True)
            loss = probability_rmse_loss_compiled(
                weight, x_batch, target_batch, softmax_scale_t
            )
            loss.backward()
            grad = weight.grad
            if grad is None:
                raise RuntimeError("compile warmup did not produce a weight gradient")
            with torch.no_grad():
                if variant == "AdamH_H1":
                    adam_h1_step_compiled(
                        weight,
                        grad,
                        torch.zeros_like(weight),
                        torch.zeros_like(weight),
                        step_t,
                        lr_t,
                        beta1_t,
                        beta2_t,
                        eps_t,
                        lr_factor_t,
                    )
                elif variant == "SGDH_H1":
                    sgdh_h1_step_compiled(
                        weight,
                        grad,
                        torch.zeros_like(weight),
                        lr_t,
                        momentum_t,
                        lr_factor_t,
                    )

    eval_shapes = {min(config.eval_batch_size, x.size(0))}
    final_eval_batch = x.size(0) % config.eval_batch_size
    if final_eval_batch:
        eval_shapes.add(final_eval_batch)
    for eval_batch_size in sorted(eval_shapes):
        _ = softmax_probs_compiled(
            initial_weight.detach(), x[:eval_batch_size], softmax_scale_t
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - start_time
    print(
        "COMPILE_WARMUP_DONE "
        + json.dumps({"elapsed_sec": elapsed_sec}, sort_keys=True),
        flush=True,
    )


def train_one_run(
    hparams: dict[str, Any],
    x: torch.Tensor,
    noisy_target_probs: torch.Tensor,
    clean_target_probs: torch.Tensor,
    initial_weight: torch.Tensor,
    config: Config,
    device: torch.device,
    candidate_idx: int,
    num_candidates: int,
) -> dict[str, Any]:
    if config.compile and supports_compiled_run(hparams):
        return train_one_run_compiled(
            hparams,
            x,
            noisy_target_probs,
            clean_target_probs,
            initial_weight,
            config,
            device,
            candidate_idx,
            num_candidates,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    model = FixedScaleSoftmax(config, initial_weight).to(device)
    optimizer = make_optimizer(model, hparams)
    batch_generator = torch.Generator(device=device)
    batch_generator.manual_seed(config.seed + 10_000)

    last_loss = float("nan")
    base_lr = float(hparams["lr"])
    steps = int(hparams.get("steps", config.steps))
    batch_size = int(hparams.get("batch_size", config.batch_size))
    sample_mode = str(hparams.get("sample_mode", "randint"))
    sample_state: dict[str, Any] = {}
    training_start_time = time.time()
    for step in range(steps):
        lr_factor = lr_schedule_factor(step, steps, hparams)
        if isinstance(optimizer, torch.optim.Optimizer):
            set_optimizer_lr(optimizer, base_lr * lr_factor)
        batch_idx = next_batch_indices(
            x.size(0),
            batch_size,
            device=device,
            generator=batch_generator,
            sample_state=sample_state,
            sample_mode=sample_mode,
        )
        loss = probability_rmse_loss(model(x[batch_idx]), noisy_target_probs[batch_idx])

        optimizer.zero_grad()
        loss.backward()
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.step()
        else:
            optimizer.step(lr_factor)

        last_loss = float(loss.detach().item())
        should_log = config.log_every > 0 and (
            (step + 1) % config.log_every == 0 or step + 1 == steps
        )
        if should_log:
            print(
                f"candidate {candidate_idx + 1:04d}/{num_candidates:04d} "
                f"{hparams['variant']} step {step + 1:05d}/{steps:05d} "
                f"loss_rmse={last_loss:.8g} lr={base_lr * lr_factor:.6g}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_elapsed_sec = time.time() - training_start_time
    clean_rmse = evaluate_clean_rmse(model, x, clean_target_probs, config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_sec = time.time() - start_time
    peak_allocated_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    peak_reserved_bytes = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
    summary = {
        **hparams,
        "candidate_idx": candidate_idx,
        "steps": steps,
        "batch_size": batch_size,
        "num_samples": int(hparams.get("num_samples", steps * batch_size)),
        "actual_samples": steps * batch_size,
        "compiled": False,
        "final_train_loss": last_loss,
        "clean_rmse": clean_rmse,
        "clean_train_rmse": clean_rmse,
        "target_met": clean_rmse <= config.target_rmse,
        "weight_row_norm_mean": float(model.weight.detach().norm(dim=1).mean().item()),
        "weight_row_norm_std": float(model.weight.detach().norm(dim=1).std().item()),
        "duration_sec": elapsed_sec,
        "training_elapsed_sec": training_elapsed_sec,
        "elapsed_sec": elapsed_sec,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_allocated_mib": peak_allocated_bytes / 1024**2,
        "peak_reserved_bytes": peak_reserved_bytes,
        "peak_reserved_mib": peak_reserved_bytes / 1024**2,
    }
    print(f"RUN_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


def main() -> None:
    config = parse_args()
    validate_variants(config.optimizer_variants)
    if config.hparam_mode == "full":
        unsupported = sorted(set(config.optimizer_variants) - set(FULL_SEARCH_VARIANTS))
        if unsupported:
            raise ValueError(
                "--hparam-mode full supports only: " + ", ".join(FULL_SEARCH_VARIANTS)
            )
    if config.step_mode == "min" and config.hparam_mode != "best":
        raise ValueError("--step-mode min requires --hparam-mode best")
    if config.shared_shape and config.step_mode != "config":
        raise ValueError("--shared-shape requires --step-mode config")
    if config.softmax_scale != 10.0:
        raise ValueError("this experiment fixes --softmax-scale to 10.0")
    if config.ground_truth_softmax_scale != 10.0:
        raise ValueError("this experiment fixes --ground-truth-softmax-scale to 10.0")

    torch.set_float32_matmul_precision("high")
    set_seed(config.seed)
    device = resolve_device(config.device)

    print(
        "CONFIG "
        + json.dumps({**asdict(config), "device": str(device)}, sort_keys=True),
        flush=True,
    )
    x, _, clean_target_probs, noisy_target_probs = build_problem(config, device)
    initial_weight = normalize_rows(
        torch.randn(config.num_classes, config.input_dim, device=device)
    )
    warmup_compiled_training(config, x, noisy_target_probs, initial_weight, device)

    summaries: list[dict[str, Any]] = []
    best_by_variant: dict[str, dict[str, Any]] = {}
    if config.hparam_mode == "full":
        effective_search_rounds = 1
        effective_runs_per_variant = 1
        total_candidates = len(build_full_search_hparams(config))
    else:
        effective_search_rounds = (
            1 if config.hparam_mode == "best" else config.search_rounds
        )
        effective_runs_per_variant = (
            1 if config.hparam_mode == "best" else config.runs_per_variant
        )
        total_candidates = (
            len(config.optimizer_variants)
            * effective_runs_per_variant
            * effective_search_rounds
        )

    print(
        "SEARCH_PLAN "
        + json.dumps(
            {
                "hparam_mode": config.hparam_mode,
                "num_candidates": total_candidates,
                "optimizer_variants": config.optimizer_variants,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for round_idx in range(effective_search_rounds):
        print(
            f"SEARCH_ROUND {json.dumps({'round': round_idx + 1, 'search_rounds': effective_search_rounds})}",
            flush=True,
        )
        if config.hparam_mode == "full":
            candidate_specs = build_full_search_hparams(config)
        else:
            candidate_specs = []
            for variant in config.optimizer_variants:
                if config.hparam_mode == "best":
                    hparams = saved_hparams(variant, sample_idx=0)
                    hparams["round"] = round_idx + 1
                    if config.step_mode == "min":
                        hparams = attach_min_steps(hparams)
                    else:
                        hparams.pop("steps", None)
                    if config.shared_shape:
                        hparams.pop("batch_size", None)
                    candidate_specs.append(hparams)
                    continue

                variant_seed = sum(
                    (idx + 1) * ord(ch) for idx, ch in enumerate(variant)
                )
                rng = random.Random(config.seed + 100_003 * round_idx + variant_seed)
                previous_best = best_by_variant.get(variant)
                for sample_idx in range(effective_runs_per_variant):
                    hparams = sample_hparams(variant, rng, previous_best, round_idx)
                    hparams["round"] = round_idx + 1
                    hparams["sample_idx"] = sample_idx
                    candidate_specs.append(hparams)

        for hparams in candidate_specs:
            candidate_idx = len(summaries)
            print(
                "RUN_START "
                + json.dumps(
                    {
                        **hparams,
                        "candidate_idx": candidate_idx,
                        "num_candidates": total_candidates,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            summary = train_one_run(
                hparams,
                x,
                noisy_target_probs,
                clean_target_probs,
                initial_weight,
                config,
                device,
                candidate_idx,
                total_candidates,
            )
            summaries.append(summary)
            variant = str(summary["variant"])
            best = best_by_variant.get(variant)
            if best is None or float(summary["clean_train_rmse"]) < float(
                best["clean_train_rmse"]
            ):
                best_by_variant[variant] = summary

        round_best = {
            variant: best_by_variant[variant] for variant in config.optimizer_variants
        }
        print(f"ROUND_BEST {json.dumps(round_best, sort_keys=True)}", flush=True)

    summaries_by_rmse = sorted(
        summaries, key=lambda item: float(item["clean_train_rmse"])
    )
    best_ordered = {
        variant: best_by_variant[variant] for variant in config.optimizer_variants
    }
    normalized_best = {
        variant: best_by_variant[variant]
        for variant in config.optimizer_variants
        if variant not in BASELINE_VARIANTS
    }

    print(f"BEST_BY_VARIANT {json.dumps(best_ordered, sort_keys=True)}", flush=True)
    print(
        f"BEST_NORMALIZED_VARIANTS {json.dumps(normalized_best, sort_keys=True)}",
        flush=True,
    )
    print(f"FINAL_SUMMARY {json.dumps(summaries_by_rmse, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
