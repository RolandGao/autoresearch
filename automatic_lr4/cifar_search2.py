"""
CIFAR-10 learning-rate and momentum search built around the Airbench94 Muon model.
"""

import copy
import os
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from math import ceil, floor, isclose, log, log10
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn

TrainingState = dict[str, Any]
Hparams = dict[str, float]
Point = tuple[int | str, ...]


class SearchResult(dict[str, Any]):
    """Result mapping shared by training, candidate, and interval search phases."""


USE_CUDNN_BENCHMARK = False
USE_TF32 = True

USE_COMPILED_MODEL = True
USE_COMPILED_MUON = True
MUON_DTYPE = torch.bfloat16
TRAINING_SEED = 0


def configure_torch_backends():
    torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
    use_tf32 = USE_TF32 and torch.cuda.is_available() and torch.cuda.is_tf32_supported()
    fp32_precision = "tf32" if use_tf32 else "ieee"
    torch.backends.cudnn.fp32_precision = fp32_precision
    torch.backends.cuda.matmul.fp32_precision = fp32_precision


def maybe_compile_model(model):
    return torch.compile(model, dynamic=False) if USE_COMPILED_MODEL else model


def set_training_seed():
    torch.manual_seed(TRAINING_SEED)


def zeropower_via_newtonschulz5(G, steps=3, eps=0):
    """Approximate a matrix's zeroth power with a quintic Newton-Schulz iteration."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(MUON_DTYPE)
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


if USE_COMPILED_MUON:
    zeropower_via_newtonschulz5 = torch.compile(
        zeropower_via_newtonschulz5, dynamic=False
    )


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.mul_(len(p) ** 0.5 / p.norm())  # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(
                    g.shape
                )  # whiten the update
                p.add_(update, alpha=-lr)  # take a step
        return loss


CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))


def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-r, r + 1, size=(len(images), 2), device=images.device)
    y = torch.arange(crop_size, device=images.device).view(1, crop_size, 1)
    x = torch.arange(crop_size, device=images.device).view(1, 1, crop_size)
    y_idx = y + (r + shifts[:, 0]).view(-1, 1, 1)
    x_idx = x + (r + shifts[:, 1]).view(-1, 1, 1)
    gathered_rows = images.gather(
        2, y_idx[:, None, :, :].expand(-1, images.size(1), -1, images.size(-1))
    )
    return gathered_rows.gather(
        3, x_idx[:, None, :, :].expand(-1, images.size(1), crop_size, crop_size)
    )


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels}, data_path)

        data = torch.load(
            data_path,
            map_location=torch.device("cuda"),
            weights_only=True,
        )
        self.images, self.labels = data["images"], data["labels"]
        # It's faster to load+process uint8 data than to load preprocessed data
        self.images = (
            (self.images.float() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        # Cache image processing results after the first epoch.
        self.proc_images = {}
        self.epoch = 0

        self.aug = aug or {}
        unknown_aug = self.aug.keys() - {"flip", "translate"}
        if unknown_aug:
            raise ValueError(f"Unrecognized augmentation keys: {sorted(unknown_aug)}")

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def _ensure_proc_images(self):
        if "norm" in self.proc_images:
            return
        images = self.proc_images["norm"] = self.normalize(self.images)
        if self.aug.get("flip", False):
            images = self.proc_images["flip"] = batch_flip_lr(images)
        pad = self.aug.get("translate", 0)
        if pad > 0:
            self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

    def normalized_images(self):
        self._ensure_proc_images()
        return self.proc_images["norm"]

    def prepare_epoch(self):
        self._ensure_proc_images()
        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity
        # relative to random flipping.
        if self.aug.get("flip", False) and self.epoch % 2 == 1:
            images = images.flip(-1)
        self.epoch += 1
        return images

    def epoch_indices(self, images):
        index_fn = torch.randperm if self.shuffle else torch.arange
        return index_fn(len(images), device=images.device)

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )


# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight
        torch.nn.init.dirac_(w[: w.size(1)])


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(
            3, whiten_width, whiten_kernel_size, padding=0, bias=True
        )
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)

    @torch.no_grad()
    def reset(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, BatchNorm, nn.Linear)):
                m.reset_parameters()
        w = self.head.weight
        w *= 1 / w.std()

    @torch.no_grad()
    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = (
            train_images.unfold(2, h, 1)
            .unfold(3, w, 1)
            .transpose(1, 3)
            .reshape(-1, c, h, w)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1, c, h, w) / torch.sqrt(
            eigenvalues.view(-1, 1, 1, 1) + eps
        )
        self.whiten.weight.copy_(torch.cat((eigenvectors_scaled, -eigenvectors_scaled)))

    def forward(self, x):
        x = F.conv2d(x, self.whiten.weight, self.whiten.bias)
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


def format_log_value(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def log_event(name, **fields):
    field_parts = [f"{key}={format_log_value(value)}" for key, value in fields.items()]
    print(" ".join([name, *field_parts]), flush=True)


def log_train_hparams(
    run,
    interval_index,
    start_step,
    completed_steps,
    total_steps,
    result,
    muon_nesterov,
):
    log_event(
        "train_hparams",
        run=run,
        interval=interval_index,
        start_step=start_step,
        completed_steps=completed_steps,
        total_steps=total_steps,
        **hparam_log_fields(result),
        muon_nesterov=muon_nesterov,
    )


def log_train_loss(run, interval_index, step, loss):
    log_event("train_loss", run=run, interval=interval_index, step=step, loss=loss)


def log_interval_boundary_eval(run, interval_index, step, total_steps, tta_val_acc):
    log_event(
        "interval_boundary_eval",
        run=run,
        interval=interval_index,
        step=step,
        total_steps=total_steps,
        tta_val_acc=tta_val_acc,
    )


def hparam_log_fields(values):
    return {name: values[name] for name in SEARCH_HPARAMS if name in values}


def copy_hparams(source: SearchResult) -> Hparams:
    return {key: source[key] for key in SEARCH_HPARAMS}


def finite(value):
    return bool(torch.isfinite(torch.as_tensor(value)).item())


RUN_SUMMARY_SPECS = """
Batch size:              %d|batch_size
Train epochs:            %.3g|train_epochs
Train steps:             %d|train_steps
N steps:                 %d|n_steps
M cooldown steps:        %d|m_steps
Search hparams:          %s|search_names_text
Cooldown hparams:        %s|cooldown_search_names_text
Muon nesterov:           %s|muon_nesterov
Full grid search:        %s|full_grid_search
Search cooldown of main: %s|search_cooldown_of_main
""".strip().splitlines()
RUN_FOOTER_SPECS = "\nVal acc:             %.4f|val_acc\nTTA val acc:         %.4f|tta_val_acc\nRun seconds:         %.3f|run_seconds\n".strip().splitlines()
RUN_RESULT_FIELDS = tuple(spec.rsplit("|", 1)[1] for spec in RUN_SUMMARY_SPECS)


def print_summary_specs(specs, result):
    for spec in specs:
        format_string, key = spec.split("|")
        print(format_string % result[key])


def log_run_summary(result):
    print_summary_specs(RUN_SUMMARY_SPECS, result)
    print_summary_specs(RUN_FOOTER_SPECS, result)


def hparam_text(values):
    return " ".join(
        f"{name}={format_log_value(values[name])}"
        for name in SEARCH_HPARAMS
        if name in values
    )


def log_line(text):
    print(text, flush=True)


def log_main_hparams(result):
    prefix = f"main hparams: {hparam_text(result)}"
    metrics = [f"main={format_log_value(result['main_tta_val_acc'])}"]
    cooldown_result = result.get("cooldown_result")
    if cooldown_result is not None:
        metrics.append(
            f"best_cooldown={format_log_value(cooldown_result['tta_val_acc'])}"
        )
    log_line(f"{prefix} {', '.join(metrics)}")


def log_candidate_result(result):
    prefix = hparam_text(result)
    log_line(f"{prefix} -> tta_val_acc={format_log_value(result['tta_val_acc'])}")


def log_candidate_results(results):
    for result in results:
        log_candidate_result(result)


def log_search_path(path):
    for step, result in enumerate(path):
        prefix = f"search_path step={step} {hparam_text(result)}"
        log_line(f"{prefix} tta_val_acc={format_log_value(result['tta_val_acc'])}")


def infer(model, loader, tta_level=0):
    """Run inference with optional mirroring and one-pixel translation TTA."""

    def infer_basic(inputs, net):
        return net(inputs)

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,) * 4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [
            infer_mirror(inputs_translate, net)
            for inputs_translate in inputs_translate_list
        ]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.train()
    test_images = loader.normalized_images()
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.inference_mode():
        return torch.cat(
            [infer_fn(inputs, model) for inputs in test_images.split(loader.batch_size)]
        )


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def evaluate_tta_val_acc(model, loader):
    return evaluate(model, loader, tta_level=2)


TRAIN_EPOCHS = 8
LABEL_SMOOTHING = 0.2
SEARCH_STEP_CONFIGS: list[tuple[int, int]] = [(40, 40)]
PRINT_OUTPUT_FILENAME = "cifar_search_coarse_to_fine.log"
LR_SEARCH_SIG_FIGS = 2
SMALL_LR_THRESHOLD_STEPS = 3
LR_ZERO_STATE = "zero"
MOMENTUM_SEARCH_VALUES = [round(i / 10, 1) for i in range(10)] + [0.95, 0.99]
INITIAL_MOMENTUM = 0.6
MUON_NESTEROV = False
FULL_GRID_SEARCH = False
SEARCH_COOLDOWN_OF_MAIN = False
INITIAL_MAIN_LR_DROPOFF_MARGIN = 0.02
INITIAL_MAIN_LR_SIDE_STEPS = 20
FULL_GRID_SEARCH_HPARAMS = ("whiten_bias_lr", "bn_bias_lr", "head_lr")
FULL_GRID_SEARCH_STATES = tuple(range(0, -21, -1))
RUN_CONFIGS: list[dict[str, Any]] = [
    dict(
        batch_size=2000,
        full_grid_search=FULL_GRID_SEARCH,
        search_cooldown_of_main=SEARCH_COOLDOWN_OF_MAIN,
    ),
]
SEARCH_HPARAM_ORDER = (
    "muon_lr",
    "muon_momentum",
    "whiten_bias_lr",
    "bn_bias_lr",
    "head_lr",
)
SEARCH_HPARAM_GROUPS = (
    ("muon_lr", "muon_momentum"),
    ("head_lr",),
    ("whiten_bias_lr",),
    ("bn_bias_lr",),
)


@dataclass(frozen=True)
class SearchHparam:
    kind: str
    initial_value: float
    search: bool = False
    cooldown_search: bool = False
    factor: float = 1
    precision: float = 1
    values: tuple[float, ...] = ()


def log_lr_hparam(initial_value=1):
    return SearchHparam(
        "log_lr", initial_value, search=True, cooldown_search=True, factor=0.6
    )


SEARCH_HPARAMS = {
    "muon_lr": log_lr_hparam(1.0),
    "muon_momentum": SearchHparam(
        "choice",
        INITIAL_MOMENTUM,
        search=True,
        values=tuple(MOMENTUM_SEARCH_VALUES),
    ),
    **{name: log_lr_hparam() for name in ("whiten_bias_lr", "bn_bias_lr", "head_lr")},
}


@dataclass
class RunConfig:
    model: Any
    run: int
    batch_size: int
    n_steps: int
    m_steps: int
    full_grid_search: bool
    search_cooldown_of_main: bool
    train_epochs: float
    initial_muon_momentum: float
    muon_nesterov: bool
    initial_whiten_bias_lr: float
    initial_bn_bias_lr: float
    initial_head_lr: float
    train_steps: int = 0
    search_names_text: str = ""
    cooldown_search_names_text: str = ""


def rounded_lr(value):
    if value == 0:
        return 0.0
    return round(value, LR_SEARCH_SIG_FIGS - 1 - floor(log10(abs(value))))


def lr_from_k(k, initial_value, factor, precision: float = 1):
    return rounded_lr(initial_value * factor ** (precision * k))


def nearest_lr_k(lr, initial_value, factor, precision: float = 1):
    if lr <= 0:
        raise ValueError(f"LR must be positive, got {lr}")
    if precision <= 0:
        raise ValueError(f"LR precision must be positive, got {precision}")
    return int(round(log(lr / initial_value) / log(factor) / precision))


def nearest_momentum_index(momentum, values):
    for index, search_momentum in enumerate(values):
        if isclose(momentum, search_momentum, rel_tol=0.0, abs_tol=1e-12):
            return index
    raise ValueError(f"momentum must be in {values}, got {momentum}")


def active_search_hparam_names():
    return search_hparam_names("search")


def cooldown_search_hparam_names():
    return search_hparam_names("cooldown_search")


def search_hparam_names(flag, allowed=None):
    return tuple(
        name
        for name in SEARCH_HPARAM_ORDER
        if (allowed is None or name in allowed) and getattr(SEARCH_HPARAMS[name], flag)
    )


def cooldown_search_hparam_names_for_step(step_name):
    for group in SEARCH_HPARAM_GROUPS:
        if step_name in group:
            return search_hparam_names("cooldown_search", group)
    raise ValueError(f"Unrecognized search hparam group for: {step_name}")


def hparam_from_state(name, state):
    spec = SEARCH_HPARAMS[name]
    if spec.kind == "log_lr":
        if state == LR_ZERO_STATE:
            return 0.0
        return lr_from_k(state, spec.initial_value, spec.factor, spec.precision)
    if spec.kind == "choice":
        return spec.values[state]
    raise ValueError(f"Unrecognized hparam kind: {spec.kind}")


def nearest_hparam_state(name, value):
    spec = SEARCH_HPARAMS[name]
    if spec.kind == "log_lr":
        if value == 0:
            return LR_ZERO_STATE
        return nearest_lr_k(value, spec.initial_value, spec.factor, spec.precision)
    if spec.kind == "choice":
        return nearest_momentum_index(value, spec.values)
    raise ValueError(f"Unrecognized hparam kind: {spec.kind}")


def initial_hparams_from_cfg(cfg):
    return {
        name: getattr(cfg, f"initial_{name}", spec.initial_value)
        for name, spec in SEARCH_HPARAMS.items()
    }


def point_from_hparams(hparams, search_names):
    return tuple(nearest_hparam_state(name, hparams[name]) for name in search_names)


def point_to_hparams(point, search_names, fixed_hparams):
    hparams = dict(fixed_hparams)
    hparams.update(
        (name, hparam_from_state(name, state))
        for name, state in zip(search_names, point)
    )
    return hparams


def format_hparam_names(names):
    return ",".join(names) if names else "none"


class FullDatasetBatchStream:
    def __init__(self, loader):
        self.loader = loader
        self.images: torch.Tensor | None = None
        self.indices: torch.Tensor | None = None
        self.batch_index = 0

    def _rng_state(self):
        return dict(
            cpu=torch.random.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

    def _load_rng_state(self, state):
        torch.random.set_rng_state(state["cpu"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    def state_dict(self):
        return dict(
            loader_epoch=self.loader.epoch,
            images=self.images,
            indices=self.indices,
            batch_index=self.batch_index,
            rng=self._rng_state(),
        )

    def load_state_dict(self, state):
        self.loader.epoch = state["loader_epoch"]
        self.images = state["images"]
        self.indices = state["indices"]
        self.batch_index = state["batch_index"]
        self._load_rng_state(state["rng"])

    def _prepare_epoch(self):
        self.images = self.loader.prepare_epoch()
        self.indices = self.loader.epoch_indices(self.images)
        self.batch_index = 0

    def next_batch(self):
        if self.images is None or self.batch_index >= len(self.loader):
            self._prepare_epoch()
        assert self.images is not None and self.indices is not None
        start = self.batch_index * self.loader.batch_size
        end = (self.batch_index + 1) * self.loader.batch_size
        idxs = self.indices[start:end]
        self.batch_index += 1
        return self.images[idxs], self.loader.labels[idxs]


@dataclass
class SearchContext:
    model: nn.Module
    muon_nesterov: bool
    optimizers: tuple[torch.optim.Optimizer, ...]
    sgd_optimizer: torch.optim.Optimizer
    muon_optimizer: Muon
    batch_stream: FullDatasetBatchStream
    test_loader: CifarLoader
    cooldown_steps: int
    total_steps: int
    search_names: tuple[str, ...]
    fixed_hparams: dict[str, float]
    full_grid_search: bool
    search_cooldown_of_main: bool


def make_optimizers(model, hparams, muon_nesterov):
    filter_params = [p for p in model.parameters() if p.ndim == 4 and p.requires_grad]
    sgd_params = {
        "whiten_bias_lr": [model.whiten.bias],
        "bn_bias_lr": [
            p
            for name, p in model.named_parameters()
            if "norm" in name and p.requires_grad
        ],
        "head_lr": [model.head.weight],
    }
    param_configs = [
        dict(params=params, lr=hparams[name], weight_decay=0, lr_name=name)
        for name, params in sgd_params.items()
    ]
    sgd_optimizer = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    muon_optimizer = Muon(
        filter_params,
        lr=hparams["muon_lr"],
        momentum=hparams["muon_momentum"],
        nesterov=muon_nesterov,
    )
    return sgd_optimizer, muon_optimizer


def set_muon_hparams(muon_optimizer, muon_lr, muon_momentum, muon_nesterov):
    for group in muon_optimizer.param_groups:
        group["lr"] = muon_lr
        group["momentum"] = muon_momentum
        group["nesterov"] = muon_nesterov


def set_sgd_lrs(ctx, hparams):
    for group in ctx.sgd_optimizer.param_groups:
        group["lr"] = hparams[group["lr_name"]]


def snapshot_training_state(model, optimizers, batch_stream):
    return dict(
        model={
            name: value.detach().clone() for (name, value) in model.state_dict().items()
        },
        optimizers=[copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers],
        batch_stream=batch_stream.state_dict(),
    )


def load_training_state(model, optimizers, batch_stream, state):
    model.load_state_dict(state["model"])
    for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
        # Optimizer state tensors are installed by reference and then mutated by
        # optimizer.step(), so every replay/candidate needs its own copy.
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    batch_stream.load_state_dict(state["batch_stream"])
    model.zero_grad(set_to_none=True)


def load_context_state(ctx, state):
    load_training_state(ctx.model, ctx.optimizers, ctx.batch_stream, state)


def train_one_step(ctx, hparams):
    set_muon_hparams(
        ctx.muon_optimizer,
        hparams["muon_lr"],
        hparams["muon_momentum"],
        ctx.muon_nesterov,
    )
    set_sgd_lrs(ctx, hparams)
    inputs, labels = ctx.batch_stream.next_batch()
    ctx.model.train()
    outputs = ctx.model(inputs)
    loss = F.cross_entropy(
        outputs, labels, label_smoothing=LABEL_SMOOTHING, reduction="mean"
    )
    loss.backward()
    for optimizer in ctx.optimizers:
        optimizer.step()
    ctx.model.zero_grad(set_to_none=True)
    return loss.item()


def train_interval(
    ctx,
    hparams: Hparams,
    interval_steps,
    capture_end_state=True,
    capture_step_metrics=False,
) -> SearchResult:
    hparams = dict(hparams)
    hparams["muon_lr"] = rounded_lr(hparams["muon_lr"])
    losses = []
    last_train_loss = float("inf")
    completed_steps = 0
    for _ in range(interval_steps):
        step_loss = train_one_step(ctx, hparams)
        last_train_loss = step_loss
        if capture_step_metrics:
            losses.append(step_loss)
        completed_steps += 1
        if not finite(step_loss):
            break
    result = SearchResult(
        last_train_loss=last_train_loss,
        completed_steps=completed_steps,
    )
    result.update(hparams)
    if capture_step_metrics:
        result["losses"] = losses
    if capture_end_state:
        result["end_state"] = snapshot_training_state(
            ctx.model, ctx.optimizers, ctx.batch_stream
        )
    return result


def interval_result_loss(result):
    return result["last_train_loss"]


def completed_finite_interval(result, steps):
    return result["completed_steps"] == steps and finite(interval_result_loss(result))


def interval_tta_val_acc(ctx, result, steps):
    if completed_finite_interval(result, steps):
        return evaluate_tta_val_acc(ctx.model, ctx.test_loader)
    return float("-inf")


def point_sort_key(point, search_names):
    values = tuple(
        hparam_from_state(name, state) for name, state in zip(search_names, point)
    )
    log_distance = sum(
        (
            float("inf")
            if state == LR_ZERO_STATE
            else abs(SEARCH_HPARAMS[name].precision * state)
        )
        for name, state in zip(search_names, point)
        if SEARCH_HPARAMS[name].kind == "log_lr"
    )
    return log_distance, values, tuple(map(str, point))


def better_point(point, incumbent_point, results_by_point, search_names):
    if incumbent_point is None:
        return True
    tta_val_acc = results_by_point[point]["tta_val_acc"]
    incumbent_tta_val_acc = results_by_point[incumbent_point]["tta_val_acc"]
    return (-tta_val_acc, point_sort_key(point, search_names)) < (
        -incumbent_tta_val_acc,
        point_sort_key(incumbent_point, search_names),
    )


def point_with_state(point, index, state):
    return point[:index] + (state,) + point[index + 1 :]


def neighbor_states(name, state, step=1):
    spec = SEARCH_HPARAMS[name]
    if spec.kind == "log_lr":
        if state == LR_ZERO_STATE:
            return []
        return [state - step, state + step]
    if spec.kind == "choice":
        states = []
        if state - step >= 0:
            states.append(state - step)
        if state + step < len(spec.values):
            states.append(state + step)
        return states
    raise ValueError(f"Unrecognized hparam kind: {spec.kind}")


def lower_value_first(points, search_names, index):
    name = search_names[index]
    return sorted(points, key=lambda point: hparam_from_state(name, point[index]))


def ordered_search_indexes(search_names):
    return [
        search_names.index(name) for name in SEARCH_HPARAM_ORDER if name in search_names
    ]


@dataclass
class HparamSearchPolicy:
    search_names: tuple[str, ...]
    evaluate: Any
    results_by_point: dict[Point, SearchResult]
    full_grid_search: bool = False
    finalize_direction_best: bool = True

    def point_in_direction(self, point, index, delta):
        name = self.search_names[index]
        spec = SEARCH_HPARAMS[name]
        if point[index] == LR_ZERO_STATE:
            return None
        state = point[index] + delta
        if spec.kind == "choice" and (state < 0 or state >= len(spec.values)):
            return None
        return point_with_state(point, index, state)

    def moves_toward_smaller_lr(self, middle_point, index, neighbor_point):
        name = self.search_names[index]
        if SEARCH_HPARAMS[name].kind != "log_lr":
            return False
        middle_lr = hparam_from_state(name, middle_point[index])
        neighbor_lr = hparam_from_state(name, neighbor_point[index])
        return neighbor_lr < middle_lr

    def point_tta_val_acc(self, point):
        return self.results_by_point[point]["tta_val_acc"]

    def is_below_initial_main_lr_cutoff(self, point, max_tta_val_acc):
        tta_val_acc = self.point_tta_val_acc(point)
        if not finite(tta_val_acc):
            return True
        if not finite(max_tta_val_acc):
            return False
        return tta_val_acc < max_tta_val_acc - INITIAL_MAIN_LR_DROPOFF_MARGIN

    def is_above_middle(self, point, middle_point):
        point_acc = self.point_tta_val_acc(point)
        middle_acc = self.point_tta_val_acc(middle_point)
        return finite(point_acc) and finite(middle_acc) and point_acc > middle_acc

    def is_below(self, point, reference_point):
        point_acc = self.point_tta_val_acc(point)
        reference_acc = self.point_tta_val_acc(reference_point)
        if not finite(point_acc) or not finite(reference_acc):
            return True
        return point_acc < reference_acc

    def best_line_point(self, incumbent_point, point):
        return point if self.is_better(point, incumbent_point) else incumbent_point

    def is_better(self, point, incumbent_point):
        return better_point(
            point, incumbent_point, self.results_by_point, self.search_names
        )

    def evaluate_line_point(self, point, parent_point, index):
        self.evaluate(
            point,
            cooldown_seed_point=parent_point,
            cooldown_search_names=cooldown_search_hparam_names_for_step(
                self.search_names[index]
            ),
        )

    def should_stop_after_candidate(
        self,
        middle_point,
        current_point,
        index,
        crossed_threshold,
        small_lr_threshold_steps,
    ):
        if crossed_threshold:
            return True, small_lr_threshold_steps
        if not self.moves_toward_smaller_lr(middle_point, index, current_point):
            return False, small_lr_threshold_steps
        small_lr_threshold_steps += 1
        return (
            small_lr_threshold_steps >= SMALL_LR_THRESHOLD_STEPS,
            small_lr_threshold_steps,
        )

    def search_direction(self, middle_point, index, first_point):
        delta = first_point[index] - middle_point[index]
        current_point = first_point
        parent_point = middle_point
        best_point = middle_point
        went_above_middle = False
        small_lr_threshold_steps = 0
        while current_point is not None:
            self.evaluate_line_point(current_point, parent_point, index)
            prior_best_point = best_point
            best_point = self.best_line_point(prior_best_point, current_point)
            found_new_best_above_middle = (
                best_point == current_point
                and prior_best_point != current_point
                and self.is_above_middle(current_point, middle_point)
            )
            if found_new_best_above_middle:
                small_lr_threshold_steps = 0
                went_above_middle = True
            else:
                went_above_middle |= self.is_above_middle(
                    prior_best_point, middle_point
                )
                crossed_threshold = (
                    self.is_below(current_point, best_point)
                    if went_above_middle
                    else self.is_below(current_point, middle_point)
                )
                should_stop, small_lr_threshold_steps = (
                    self.should_stop_after_candidate(
                        middle_point,
                        current_point,
                        index,
                        crossed_threshold,
                        small_lr_threshold_steps,
                    )
                )
                if should_stop:
                    break
            next_point = self.point_in_direction(current_point, index, delta)
            parent_point = current_point
            current_point = next_point
        if went_above_middle:
            return best_point
        return middle_point

    def search_full_grid(self, middle_point, index):
        best_point = middle_point
        points = [
            point_with_state(middle_point, index, state)
            for state in FULL_GRID_SEARCH_STATES
        ]
        for point in points:
            self.evaluate_line_point(point, middle_point, index)
            best_point = self.best_line_point(best_point, point)
        return best_point

    def search_neighbor_directions(self, middle_point, index):
        name = self.search_names[index]
        neighbors = [
            point_with_state(middle_point, index, state)
            for state in neighbor_states(name, middle_point[index])
        ]
        neighbors = lower_value_first(neighbors, self.search_names, index)
        for neighbor_point in neighbors:
            direction_best_point = self.search_direction(
                middle_point, index, neighbor_point
            )
            if direction_best_point == middle_point:
                continue
            if self.finalize_direction_best:
                self.evaluate(direction_best_point)
            if self.is_better(direction_best_point, middle_point):
                return direction_best_point
        return middle_point

    def search_hparam_sweep(self, middle_point, initial_main_search=False):
        accepted_points = []
        for index in ordered_search_indexes(self.search_names):
            name = self.search_names[index]
            if initial_main_search and SEARCH_HPARAMS[name].kind == "log_lr":
                next_point = self.search_initial_main_lr(middle_point, index)
            elif (
                not initial_main_search
                and self.full_grid_search
                and name in FULL_GRID_SEARCH_HPARAMS
            ):
                next_point = self.search_full_grid(middle_point, index)
            else:
                next_point = self.search_neighbor_directions(middle_point, index)
            accept = (
                next_point != middle_point
                if initial_main_search
                else self.is_better(next_point, middle_point)
            )
            if accept:
                middle_point = next_point
                accepted_points.append(middle_point)
        return middle_point, accepted_points

    def update_max_tta_val_acc(self, max_tta_val_acc, point):
        tta_val_acc = self.point_tta_val_acc(point)
        if not finite(tta_val_acc):
            return max_tta_val_acc
        if not finite(max_tta_val_acc) or tta_val_acc > max_tta_val_acc:
            return tta_val_acc
        return max_tta_val_acc

    def probe_initial_lr_side(
        self, middle_point, index, probe_points, max_tta_val_acc, delta
    ):
        previous_point = middle_point
        current_point = self.point_in_direction(middle_point, index, delta)
        for _ in range(INITIAL_MAIN_LR_SIDE_STEPS):
            if current_point is None:
                break
            self.evaluate_line_point(current_point, previous_point, index)
            probe_points.append(current_point)
            if self.is_below_initial_main_lr_cutoff(current_point, max_tta_val_acc):
                break
            max_tta_val_acc = self.update_max_tta_val_acc(
                max_tta_val_acc, current_point
            )
            previous_point = current_point
            current_point = self.point_in_direction(current_point, index, delta)
        return max_tta_val_acc

    def is_above_final_cutoff(self, point, max_tta_val_acc):
        tta_val_acc = self.point_tta_val_acc(point)
        if not finite(tta_val_acc) or not finite(max_tta_val_acc):
            return False
        return tta_val_acc >= max_tta_val_acc - INITIAL_MAIN_LR_DROPOFF_MARGIN

    def search_initial_main_lr(self, middle_point, index):
        name = self.search_names[index]

        probe_points = [middle_point]
        max_tta_val_acc = self.point_tta_val_acc(middle_point)
        for delta in (-1, 1):
            max_tta_val_acc = self.probe_initial_lr_side(
                middle_point, index, probe_points, max_tta_val_acc, delta
            )
        candidate_points = [
            point
            for point in probe_points
            if self.is_above_final_cutoff(point, max_tta_val_acc)
        ]
        if not candidate_points:
            return middle_point
        return max(
            candidate_points,
            key=lambda point: hparam_from_state(name, point[index]),
        )

    def run(self, initial_point, initial_main_search=False):
        middle_point = initial_point
        center_path = [middle_point]
        self.evaluate(middle_point)
        if initial_main_search:
            middle_point, accepted_points = self.search_hparam_sweep(
                middle_point, initial_main_search=True
            )
            center_path.extend(accepted_points)
            return middle_point, center_path
        while True:
            next_point, accepted_points = self.search_hparam_sweep(middle_point)
            if not accepted_points:
                break
            middle_point = next_point
            center_path.extend(accepted_points)
        return middle_point, center_path


def find_best_hparam_point(
    initial_point,
    search_names,
    evaluate,
    results_by_point,
    full_grid_search=False,
    finalize_direction_best=True,
    initial_main_search=False,
):
    policy = HparamSearchPolicy(
        search_names=search_names,
        evaluate=evaluate,
        results_by_point=results_by_point,
        full_grid_search=full_grid_search,
        finalize_direction_best=finalize_direction_best,
    )
    return policy.run(initial_point, initial_main_search)


def add_best_result_fields(result, best_result):
    result.update(copy_hparams(best_result))


def tta_val_acc_improved(tta_val_acc, incumbent_tta_val_acc):
    if not finite(tta_val_acc):
        return False
    if incumbent_tta_val_acc is None or not finite(incumbent_tta_val_acc):
        return True
    return tta_val_acc > incumbent_tta_val_acc


def point_states(point, search_names):
    return dict(zip(search_names, point))


def hparam_states(hparams, names):
    return {name: nearest_hparam_state(name, hparams[name]) for name in names}


@dataclass(frozen=True)
class IntervalInfo:
    interval_index: int
    cooldown_steps: int
    cooldown_initial_states: dict[str, Any] | None
    use_cooldown: bool
    fixed_cooldown_hparams: Hparams | None
    search_candidate_cooldown: bool


@dataclass
class SegmentCandidateEvaluator:
    ctx: SearchContext
    search_names: tuple[str, ...]
    fixed_hparams: Hparams
    steps: int
    start_state: TrainingState
    interval_info: IntervalInfo | None
    results_by_point: dict[Point, SearchResult]
    candidate_evaluations: list[SearchResult]

    @property
    def fixed_cooldown_hparams(self):
        return getattr(self.interval_info, "fixed_cooldown_hparams", None)

    @property
    def search_candidate_cooldown(self):
        return bool(self.interval_info and self.interval_info.search_candidate_cooldown)

    def _requested_cooldown_names(self, cooldown_search_names):
        if self.interval_info is None or not self.interval_info.use_cooldown:
            return None
        candidate_search = (
            self.search_candidate_cooldown and cooldown_search_names is not None
        )
        if self.fixed_cooldown_hparams is not None and not candidate_search:
            return None
        return (
            cooldown_search_hparam_names()
            if cooldown_search_names is None
            else tuple(cooldown_search_names)
        )

    @staticmethod
    def _cached_result_covers(cached_result, requested_cooldown_names):
        cached_names = cached_result.get("cooldown_search_names")
        if requested_cooldown_names is None:
            return cached_names in (None, ())
        return set(requested_cooldown_names).issubset(set(cached_names or ()))

    def _cooldown_initial_states(self, cached_result, cooldown_seed_point):
        assert self.interval_info is not None
        states = dict(self.interval_info.cooldown_initial_states or {})
        if cached_result is not None:
            states.update(cached_result.get("cooldown_best_states") or {})
        if cooldown_seed_point is not None:
            seed_result = self.results_by_point.get(cooldown_seed_point)
            if seed_result is not None:
                states.update(seed_result.get("cooldown_best_states") or {})
        return states

    def _search_cooldown(
        self,
        hparams,
        cooldown_start_state,
        cached_result,
        cooldown_seed_point,
        requested_cooldown_names,
    ):
        assert self.interval_info is not None
        initial_states = self._cooldown_initial_states(
            cached_result, cooldown_seed_point
        )
        cooldown_initial_hparams = dict(hparams)
        for name in cooldown_search_hparam_names():
            initial_state = initial_states.get(name)
            if initial_state is None and name in requested_cooldown_names:
                initial_state = nearest_hparam_state(name, hparams[name])
            if initial_state is not None:
                cooldown_initial_hparams[name] = hparam_from_state(name, initial_state)

        initial_results = []
        if cached_result is not None:
            cached_cooldown = cached_result.get("cooldown_result")
            if cached_cooldown is not None:
                initial_results = cached_cooldown.get("candidate_evaluations", [])
        return search_hparam_segment(
            self.ctx,
            point_from_hparams(cooldown_initial_hparams, requested_cooldown_names),
            requested_cooldown_names,
            cooldown_initial_hparams,
            self.interval_info.cooldown_steps,
            cooldown_start_state,
            initial_results=initial_results,
        )

    def _run_fixed_cooldown(self, hparams):
        assert self.interval_info is not None
        assert self.fixed_cooldown_hparams is not None
        cooldown_hparams = dict(hparams)
        for name in cooldown_search_hparam_names():
            cooldown_hparams[name] = self.fixed_cooldown_hparams[name]
        cooldown_result = train_interval(
            self.ctx,
            cooldown_hparams,
            self.interval_info.cooldown_steps,
            capture_end_state=False,
        )
        cooldown_tta_val_acc = interval_tta_val_acc(
            self.ctx, cooldown_result, self.interval_info.cooldown_steps
        )
        best_states = hparam_states(cooldown_hparams, cooldown_search_hparam_names())
        cooldown_result.update(
            tta_val_acc=cooldown_tta_val_acc,
            best_states=best_states,
        )
        return cooldown_result, best_states

    def _evaluate_main_candidate(
        self,
        result,
        hparams,
        cached_result,
        cooldown_seed_point,
        requested_cooldown_names,
        should_evaluate_tta,
    ):
        assert self.interval_info is not None
        uses_fixed_cooldown = (
            self.interval_info.use_cooldown and self.fixed_cooldown_hparams is not None
        )
        result["cooldown_result"] = None
        result["cooldown_search_names"] = (
            requested_cooldown_names
            if requested_cooldown_names is not None
            else () if uses_fixed_cooldown else None
        )

        result["main_tta_val_acc"] = (
            evaluate_tta_val_acc(self.ctx.model, self.ctx.test_loader)
            if should_evaluate_tta
            else float("-inf")
        )

        if should_evaluate_tta and (
            requested_cooldown_names is not None or uses_fixed_cooldown
        ):
            if requested_cooldown_names is not None:
                cooldown_result = self._search_cooldown(
                    hparams,
                    result["end_state"],
                    cached_result,
                    cooldown_seed_point,
                    requested_cooldown_names,
                )
                best_states = hparam_states(
                    cooldown_result, cooldown_search_hparam_names()
                )
            else:
                cooldown_result, best_states = self._run_fixed_cooldown(hparams)
            result.update(
                cooldown_result=cooldown_result,
                cooldown_best_states=best_states,
                tta_val_acc=cooldown_result["tta_val_acc"],
            )
            should_evaluate_tta = False
        result.pop("end_state", None)
        return should_evaluate_tta

    def _log_main_candidate(self, result):
        log_main_hparams(result)
        cooldown_result = result.get("cooldown_result")
        if cooldown_result is not None and "candidate_evaluations" in cooldown_result:
            log_candidate_results(cooldown_result["candidate_evaluations"])
            log_search_path(cooldown_result["search_path"])

    def __call__(self, point, cooldown_seed_point=None, cooldown_search_names=None):
        requested_cooldown_names = self._requested_cooldown_names(cooldown_search_names)
        cached_result = self.results_by_point.get(point)
        if cached_result is not None and self._cached_result_covers(
            cached_result, requested_cooldown_names
        ):
            return cached_result

        hparams = point_to_hparams(point, self.search_names, self.fixed_hparams)
        load_context_state(self.ctx, self.start_state)
        needs_cooldown_state = requested_cooldown_names is not None
        result = train_interval(
            self.ctx,
            hparams,
            self.steps,
            capture_end_state=needs_cooldown_state,
        )
        should_evaluate_tta = completed_finite_interval(result, self.steps)
        if self.interval_info is not None:
            should_evaluate_tta = self._evaluate_main_candidate(
                result,
                hparams,
                cached_result,
                cooldown_seed_point,
                requested_cooldown_names,
                should_evaluate_tta,
            )
        if should_evaluate_tta:
            result["tta_val_acc"] = (
                evaluate_tta_val_acc(self.ctx.model, self.ctx.test_loader)
                if self.interval_info is None
                else result["main_tta_val_acc"]
            )
        else:
            result.setdefault("tta_val_acc", float("-inf"))

        self.results_by_point[point] = result
        self.candidate_evaluations.append(result)
        if self.interval_info is not None:
            self._log_main_candidate(result)
        return result


def search_hparam_segment(
    ctx,
    initial_point,
    search_names,
    fixed_hparams,
    steps,
    start_state,
    interval_info=None,
    initial_results=None,
    initial_main_search=False,
) -> SearchResult:
    candidate_evaluations = list(initial_results or [])
    results_by_point = {
        point_from_hparams(result, search_names): result
        for result in candidate_evaluations
    }
    evaluator = SegmentCandidateEvaluator(
        ctx=ctx,
        search_names=search_names,
        fixed_hparams=fixed_hparams,
        steps=steps,
        start_state=start_state,
        interval_info=interval_info,
        results_by_point=results_by_point,
        candidate_evaluations=candidate_evaluations,
    )
    best_point, center_path_points = find_best_hparam_point(
        initial_point=initial_point,
        search_names=search_names,
        evaluate=evaluator,
        results_by_point=results_by_point,
        full_grid_search=ctx.full_grid_search,
        finalize_direction_best=not evaluator.search_candidate_cooldown,
        initial_main_search=initial_main_search,
    )
    search_path = [results_by_point[point] for point in center_path_points]
    best_result = results_by_point[best_point]
    result = SearchResult(
        tta_val_acc=best_result["tta_val_acc"],
        candidate_evaluations=candidate_evaluations,
        search_path=search_path,
    )
    if interval_info is None:
        result["best_states"] = point_states(best_point, search_names)
    else:
        result.update(
            interval_index=interval_info.interval_index,
            best_point=best_point,
            main_tta_val_acc=best_result["main_tta_val_acc"],
            cooldown_result=best_result.get("cooldown_result"),
            cooldown_best_states=best_result.get("cooldown_best_states"),
        )
    add_best_result_fields(result, best_result)
    load_context_state(ctx, start_state)
    return result


@dataclass
class IntervalSearchCoordinator:
    ctx: SearchContext
    initial_point: Point
    interval_steps: int
    interval_index: int
    interval_start_step: int
    cooldown_initial_states: dict[str, Any] | None = None
    cooldown_steps: int = field(init=False)
    use_cooldown: bool = field(init=False)
    start_state: TrainingState = field(init=False)

    def __post_init__(self):
        remaining_steps = (
            self.ctx.total_steps - self.interval_start_step - self.interval_steps
        )
        self.cooldown_steps = min(self.ctx.cooldown_steps, remaining_steps)
        self.use_cooldown = self.cooldown_steps > 0
        self.start_state = snapshot_training_state(
            self.ctx.model, self.ctx.optimizers, self.ctx.batch_stream
        )

    def evaluate_main_point(self, main_point):
        hparams = point_to_hparams(
            main_point, self.ctx.search_names, self.ctx.fixed_hparams
        )
        load_context_state(self.ctx, self.start_state)
        result = train_interval(
            self.ctx,
            hparams,
            self.interval_steps,
            capture_end_state=False,
        )
        main_tta_val_acc = interval_tta_val_acc(self.ctx, result, self.interval_steps)
        result.update(
            interval_index=self.interval_index,
            best_point=main_point,
            main_tta_val_acc=main_tta_val_acc,
            tta_val_acc=main_tta_val_acc,
            cooldown_result=None,
            cooldown_best_states=None,
        )
        load_context_state(self.ctx, self.start_state)
        return result

    def search_main_phase(
        self,
        initial_main_point,
        fixed_cooldown_hparams=None,
        cooldown_initial_states=None,
        search_candidate_cooldown=False,
    ):
        initial_main_search = (
            self.interval_index == 0
            and fixed_cooldown_hparams is None
            and not search_candidate_cooldown
        )
        interval_info = IntervalInfo(
            interval_index=self.interval_index,
            cooldown_steps=self.cooldown_steps,
            cooldown_initial_states=cooldown_initial_states,
            use_cooldown=(self.use_cooldown and fixed_cooldown_hparams is not None),
            fixed_cooldown_hparams=fixed_cooldown_hparams,
            search_candidate_cooldown=search_candidate_cooldown,
        )
        return search_hparam_segment(
            self.ctx,
            initial_main_point,
            self.ctx.search_names,
            self.ctx.fixed_hparams,
            self.interval_steps,
            self.start_state,
            interval_info,
            initial_main_search=initial_main_search,
        )

    @staticmethod
    def cooldown_hparams_from_states(main_hparams, states):
        hparams = dict(main_hparams)
        states = dict(states or {})
        for name in cooldown_search_hparam_names():
            state = states.get(name)
            if state is None:
                state = nearest_hparam_state(name, main_hparams[name])
            hparams[name] = hparam_from_state(name, state)
        return hparams

    def search_cooldown_phase(self, main_hparams, initial_states):
        load_context_state(self.ctx, self.start_state)
        main_result = train_interval(
            self.ctx,
            main_hparams,
            self.interval_steps,
            capture_end_state=True,
        )
        if not completed_finite_interval(main_result, self.interval_steps):
            failed_result = SearchResult(
                best_states=dict(initial_states or {}),
                tta_val_acc=float("-inf"),
                candidate_evaluations=[],
                search_path=[],
            )
            failed_result.update(main_hparams)
            load_context_state(self.ctx, self.start_state)
            return failed_result

        cooldown_search_names = cooldown_search_hparam_names()
        cooldown_initial_hparams = self.cooldown_hparams_from_states(
            main_hparams,
            initial_states,
        )
        result = search_hparam_segment(
            self.ctx,
            point_from_hparams(cooldown_initial_hparams, cooldown_search_names),
            cooldown_search_names,
            cooldown_initial_hparams,
            self.cooldown_steps,
            main_result["end_state"],
        )
        load_context_state(self.ctx, self.start_state)
        return result

    def run(self) -> SearchResult:
        cooldown_first = self.interval_index > 0 and self.use_cooldown
        if cooldown_first:
            main_result = self.evaluate_main_point(self.initial_point)
        else:
            main_result = self.search_main_phase(self.initial_point)
            log_search_path(main_result["search_path"])
        selected_main_result = main_result
        selected_tta_val_acc = main_result["tta_val_acc"]
        selected_cooldown_states = dict(self.cooldown_initial_states or {})
        fixed_cooldown_hparams = None

        while self.use_cooldown:
            cooldown_result = self.search_cooldown_phase(
                copy_hparams(selected_main_result),
                selected_cooldown_states,
            )
            log_main_hparams(
                dict(
                    copy_hparams(selected_main_result),
                    main_tta_val_acc=selected_main_result["main_tta_val_acc"],
                    cooldown_result=cooldown_result,
                )
            )
            log_candidate_results(cooldown_result["candidate_evaluations"])
            log_search_path(cooldown_result["search_path"])
            if not tta_val_acc_improved(
                cooldown_result["tta_val_acc"], selected_tta_val_acc
            ):
                break

            selected_tta_val_acc = cooldown_result["tta_val_acc"]
            selected_cooldown_states = cooldown_result["best_states"]
            fixed_cooldown_hparams = copy_hparams(cooldown_result)

            main_result = self.search_main_phase(
                selected_main_result["best_point"],
                fixed_cooldown_hparams=fixed_cooldown_hparams,
                cooldown_initial_states=selected_cooldown_states,
                search_candidate_cooldown=self.ctx.search_cooldown_of_main,
            )
            log_search_path(main_result["search_path"])
            if not tta_val_acc_improved(
                main_result["tta_val_acc"], selected_tta_val_acc
            ):
                break

            selected_main_result = main_result
            selected_tta_val_acc = main_result["tta_val_acc"]
            if main_result["cooldown_best_states"] is not None:
                selected_cooldown_states = main_result["cooldown_best_states"]

        load_context_state(self.ctx, self.start_state)
        selected_main_hparams = copy_hparams(selected_main_result)
        actual_result = train_interval(
            self.ctx,
            selected_main_hparams,
            self.interval_steps,
            capture_end_state=False,
            capture_step_metrics=True,
        )
        result = SearchResult(
            interval_index=self.interval_index,
            best_point=selected_main_result["best_point"],
            cooldown_best_states=(
                selected_cooldown_states if fixed_cooldown_hparams is not None else None
            ),
            tta_val_acc=selected_tta_val_acc,
            main_tta_val_acc=selected_main_result["main_tta_val_acc"],
        )
        add_best_result_fields(result, selected_main_hparams)
        result.update(
            completed_steps=actual_result["completed_steps"],
            losses=actual_result["losses"],
        )
        return result


def search_interval_hparams(
    ctx,
    initial_point,
    interval_steps,
    interval_index,
    interval_start_step,
    cooldown_initial_states=None,
) -> SearchResult:
    return IntervalSearchCoordinator(
        ctx=ctx,
        initial_point=initial_point,
        interval_steps=interval_steps,
        interval_index=interval_index,
        interval_start_step=interval_start_step,
        cooldown_initial_states=cooldown_initial_states,
    ).run()


def run_full_dataset_search(cfg: RunConfig) -> dict[str, Any]:
    set_training_seed()
    search_names = active_search_hparam_names()
    initial_hparams = initial_hparams_from_cfg(cfg)
    initial_point = point_from_hparams(initial_hparams, search_names)

    train_loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=cfg.batch_size,
        aug=dict(flip=True, translate=2),
    )
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    batch_stream = FullDatasetBatchStream(train_loader)
    cfg.train_steps = ceil(cfg.train_epochs * len(train_loader))

    cfg.model.reset()
    cfg.model.init_whiten(train_loader.normalized_images()[:5000])
    sgd_optimizer, muon_optimizer = make_optimizers(
        cfg.model,
        initial_hparams,
        cfg.muon_nesterov,
    )
    optimizers = (sgd_optimizer, muon_optimizer)
    search_ctx = SearchContext(
        model=cfg.model,
        muon_nesterov=cfg.muon_nesterov,
        optimizers=optimizers,
        sgd_optimizer=sgd_optimizer,
        muon_optimizer=muon_optimizer,
        batch_stream=batch_stream,
        test_loader=test_loader,
        cooldown_steps=cfg.m_steps,
        total_steps=cfg.train_steps,
        search_names=search_names,
        fixed_hparams=initial_hparams,
        full_grid_search=cfg.full_grid_search,
        search_cooldown_of_main=cfg.search_cooldown_of_main,
    )

    last_loss = None
    interval_initial_point = initial_point
    interval_cooldown_initial_states = None
    completed_steps = 0
    interval_index = 0
    while completed_steps < cfg.train_steps:
        interval_steps = min(cfg.n_steps, cfg.train_steps - completed_steps)
        interval_result = search_interval_hparams(
            search_ctx,
            interval_initial_point,
            interval_steps,
            interval_index,
            completed_steps,
            interval_cooldown_initial_states,
        )
        interval_initial_point = interval_result["best_point"]
        if interval_result["cooldown_best_states"] is not None:
            interval_cooldown_initial_states = interval_result["cooldown_best_states"]
        interval_completed_steps = int(interval_result["completed_steps"])
        actual_losses = list(interval_result["losses"][:interval_completed_steps])
        log_train_hparams(
            cfg.run,
            interval_result["interval_index"],
            completed_steps,
            interval_completed_steps,
            cfg.train_steps,
            interval_result,
            cfg.muon_nesterov,
        )
        for local_offset, loss in enumerate(actual_losses, start=1):
            global_step = completed_steps + local_offset
            last_loss = loss
            log_train_loss(
                cfg.run, interval_result["interval_index"], global_step, loss
            )
        completed_steps += interval_completed_steps
        if (
            completed_steps < cfg.train_steps
            and last_loss is not None
            and finite(last_loss)
        ):
            boundary_tta_val_acc = evaluate_tta_val_acc(cfg.model, test_loader)
            log_interval_boundary_eval(
                cfg.run,
                interval_result["interval_index"],
                completed_steps,
                cfg.train_steps,
                boundary_tta_val_acc,
            )
        interval_index += 1
        if last_loss is not None and not finite(last_loss):
            break

    result: dict[str, Any] = {"run": cfg.run}
    result.update({name: getattr(cfg, name) for name in RUN_RESULT_FIELDS})
    result.update(
        val_acc=evaluate(cfg.model, test_loader),
        tta_val_acc=evaluate_tta_val_acc(cfg.model, test_loader),
    )
    return result


def iter_run_settings(model):
    for run, (config, steps) in enumerate(product(RUN_CONFIGS, SEARCH_STEP_CONFIGS)):
        n_steps, m_steps = steps
        batch_size = config["batch_size"]
        sgd_lr_mult = batch_size / 2000
        yield RunConfig(
            model=model,
            run=run,
            n_steps=n_steps,
            m_steps=m_steps,
            batch_size=batch_size,
            full_grid_search=config.get("full_grid_search", FULL_GRID_SEARCH),
            search_cooldown_of_main=config.get(
                "search_cooldown_of_main", SEARCH_COOLDOWN_OF_MAIN
            ),
            train_epochs=TRAIN_EPOCHS,
            initial_muon_momentum=SEARCH_HPARAMS["muon_momentum"].initial_value,
            muon_nesterov=MUON_NESTEROV,
            initial_whiten_bias_lr=SEARCH_HPARAMS["whiten_bias_lr"].initial_value
            * sgd_lr_mult,
            initial_bn_bias_lr=SEARCH_HPARAMS["bn_bias_lr"].initial_value * sgd_lr_mult,
            initial_head_lr=SEARCH_HPARAMS["head_lr"].initial_value * sgd_lr_mult,
        )


def print_run_banner(cfg):
    initial_hparams = initial_hparams_from_cfg(cfg)
    initial_muon_momentum = initial_hparams["muon_momentum"]
    log_event(
        "cifar_search_simple",
        run=cfg.run,
        batch_size=cfg.batch_size,
        train_epochs=cfg.train_epochs,
        n_steps=cfg.n_steps,
        m_steps=cfg.m_steps,
        search_hparams=cfg.search_names_text,
        cooldown_search_hparams=cfg.cooldown_search_names_text,
        muon_nesterov=cfg.muon_nesterov,
        full_grid_search=cfg.full_grid_search,
        search_cooldown_of_main=cfg.search_cooldown_of_main,
        initial_muon_lr=initial_hparams["muon_lr"],
        initial_muon_momentum=initial_muon_momentum,
        initial_muon_momentum_index=nearest_hparam_state(
            "muon_momentum", initial_muon_momentum
        ),
        initial_whiten_bias_lr=initial_hparams["whiten_bias_lr"],
        initial_bn_bias_lr=initial_hparams["bn_bias_lr"],
        initial_head_lr=initial_hparams["head_lr"],
    )


def main():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    print_output_path = os.path.join(current_time, PRINT_OUTPUT_FILENAME)
    print_output_dir = os.path.dirname(print_output_path)
    if print_output_dir:
        os.makedirs(print_output_dir, exist_ok=True)
    with (
        open(print_output_path, "w") as print_output_file,
        redirect_stdout(print_output_file),
    ):
        run_main()


def run_main():
    configure_torch_backends()
    eager_model = CifarNet().cuda()
    eager_model = eager_model.to(  # pyright: ignore[reportCallIssue]
        memory_format=torch.channels_last
    )
    model = maybe_compile_model(eager_model)
    for cfg in iter_run_settings(model):
        cfg.search_names_text = format_hparam_names(active_search_hparam_names())
        cfg.cooldown_search_names_text = format_hparam_names(
            cooldown_search_hparam_names()
        )
        print_run_banner(cfg)
        run_start_time = time.perf_counter()
        result = run_full_dataset_search(cfg)
        result["run_seconds"] = time.perf_counter() - run_start_time
        log_run_summary(result)


if __name__ == "__main__":
    main()
