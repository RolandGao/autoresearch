"""
airbench94_muon.py
Runs in 2.59 seconds on a 400W NVIDIA A100 using torch==2.4.1
Attains 94.01 mean accuracy (n=200 trials)
Descends from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""

#############################################
#                  Setup                    #
#############################################

import copy
import os
import sys
import time

with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil, floor, isclose, log, log10

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


def _cuda_capability():
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


CUDA_CAPABILITY = _cuda_capability()
IS_AMPERE_OR_NEWER = CUDA_CAPABILITY[0] >= 8
USE_CUDNN_BENCHMARK = False
USE_TF32 = IS_AMPERE_OR_NEWER

torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.backends.cuda.matmul.allow_tf32 = USE_TF32

USE_COMPILED_MUON = False
MUON_DTYPE = torch.bfloat16
TRAINING_SEED = 0


def set_training_seed():
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)


#############################################
#               Muon optimizer              #
#############################################


def zeropower_via_newtonschulz5(G, steps=3, eps=0):
    r"""
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
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
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False, orthogonalize=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            orthogonalize=orthogonalize,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if "momentum_buffer" not in state.keys():
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf

                p.data.mul_(len(p.data) ** 0.5 / p.data.norm())  # normalize the weight
                update = g
                if group["orthogonalize"]:
                    update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(
                        g.shape
                    )  # whiten the update
                p.data.add_(update, alpha=-lr)  # take a step


#############################################
#                DataLoader                 #
#############################################

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
            torch.save(
                {"images": images, "labels": labels, "classes": dset.classes}, data_path
            )

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        # It's faster to load+process uint8 data than to load preprocessed data
        self.images = (
            (self.images.float() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}  # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ["flip", "translate"], "Unrecognized key: %s" % k

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

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):

        if self.epoch == 0:
            self._ensure_proc_images()

        if self.aug.get("translate", 0) > 0:
            images = batch_crop(self.proc_images["pad"], self.images.shape[-2])
        elif self.aug.get("flip", False):
            images = self.proc_images["flip"]
        else:
            images = self.proc_images["norm"]
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get("flip", False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(
            len(images), device=images.device
        )
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Definition             #
#############################################


# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

    def train(self, mode=True):
        return super().train(True)


class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
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
        for mod in self.modules():
            mod.float()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

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
        self.whiten.weight.data[:] = torch.cat(
            (eigenvectors_scaled, -eigenvectors_scaled)
        )

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


############################################
#                 Logging                  #
############################################


def log_train_loss(
    run, step, total_steps, loss, head_lr, muon_lr, muon_momentum, muon_nesterov
):
    print(
        f"train_loss run={run} step={step}/{total_steps} "
        f"loss={loss:.4f} head_lr={head_lr:.6g} "
        f"muon_lr={muon_lr:.6g} muon_momentum={format_momentum(muon_momentum)} "
        f"muon_nesterov={muon_nesterov}",
        flush=True,
    )


def format_k(k):
    return "%g" % k


def format_optional_lr(lr):
    return "none" if lr is None else "%.6g" % lr


def format_momentum(momentum):
    return "%g" % momentum


def format_optional_momentum(momentum):
    return "none" if momentum is None else format_momentum(momentum)


def log_interval_lr_landscape(results_by_point):
    for interval_point in sorted(results_by_point):
        interval_result = results_by_point[interval_point]
        cooldown_result = interval_result["cooldown_result"]
        print(
            "interval_muon_lr=%.6g interval_muon_momentum=%s "
            "interval_muon_nesterov=%s interval_loss=%.6f"
            % (
                interval_result["muon_lr"],
                format_momentum(interval_result["muon_momentum"]),
                interval_result["muon_nesterov"],
                interval_result["interval_final_loss"],
            ),
            flush=True,
        )
        if cooldown_result is None:
            print(
                "cooldown_muon_lr=none final_loss=%.6f"
                % (interval_result["final_loss"],),
                flush=True,
            )
            continue

        for cooldown_eval in cooldown_result["search_evaluations"]:
            print(
                "%.6g %s %s -> %.6f"
                % (
                    cooldown_eval["muon_lr"],
                    format_momentum(cooldown_eval["muon_momentum"]),
                    cooldown_eval["muon_nesterov"],
                    cooldown_eval["final_loss"],
                ),
                flush=True,
            )


def count_evaluated_configs(results_by_point):
    count = 0
    for result in results_by_point.values():
        cooldown_result = result["cooldown_result"]
        count += (
            len(cooldown_result["search_evaluations"])
            if cooldown_result is not None
            else 1
        )
    return count


def log_interval_lr_search_complete(
    best_result,
    best_cooldown_result,
    results_by_k,
):
    cooldown_muon_lr = (
        best_cooldown_result["muon_lr"] if best_cooldown_result is not None else None
    )
    cooldown_muon_momentum = (
        best_cooldown_result["muon_momentum"]
        if best_cooldown_result is not None
        else None
    )
    cooldown_muon_nesterov = (
        best_cooldown_result["muon_nesterov"]
        if best_cooldown_result is not None
        else None
    )
    print(
        "best_interval_muon_lr=%.6g "
        "best_interval_muon_momentum=%s "
        "best_interval_muon_nesterov=%s "
        "best_cooldown_muon_lr=%s "
        "best_cooldown_muon_momentum=%s "
        "best_cooldown_muon_nesterov=%s "
        "interval_loss=%.6f final_loss=%.6f evaluated_interval_configs=%d "
        "evaluated_configs=%d"
        % (
            best_result["muon_lr"],
            format_momentum(best_result["muon_momentum"]),
            best_result["muon_nesterov"],
            format_optional_lr(cooldown_muon_lr),
            format_optional_momentum(cooldown_muon_momentum),
            cooldown_muon_nesterov,
            best_result["interval_final_loss"],
            best_result["final_loss"],
            len(results_by_k),
            count_evaluated_configs(results_by_k),
        ),
        flush=True,
    )


############################################
#                Training                  #
############################################

OVERFIT_BATCH_SIZES = [500, 2000, 10000]
N_SEARCH_STEPS = [1, 2, 3, 4]
M_COOLDOWN_STEPS = [0, 1, 2, 3, 4]
MUON_ORTHOGONALIZE = [True]
OVERFIT_TRAIN_STEPS = 30
LR_SEARCH_BASE = 0.2
LR_SEARCH_FACTOR = 0.6
LR_SEARCH_SIG_FIGS = 2
LR_SEARCH_MAX_MOVES = 60
MOMENTUM_SEARCH_VALUES = [round(i / 10, 1) for i in range(10)] + [0.95, 0.99]
MUON_MOMENTUM_CONFIGS = [
    dict(
        momentum_config_name="search_momentum_fixed_nesterov_false",
        initial_momentum=0.6,
        search_momentum=True,
        muon_nesterov=False,
    ),
]
LABEL_SMOOTHING = 0.2
SGD_LR_MULTS = {
    125: 1.0,
    500: 1.0,
    2000: 0.8,
    5000: 0.8,
    10000: 0.8,
}
RUN_CONFIGS = [
    dict(
        batch_size=batch_size,
        initial_lr=LR_SEARCH_BASE,
        sgd_lr_mult=SGD_LR_MULTS[batch_size],
    )
    for batch_size in OVERFIT_BATCH_SIZES
]


def rounded_lr(value):
    if value == 0:
        return 0.0
    return round(value, LR_SEARCH_SIG_FIGS - 1 - floor(log10(abs(value))))


def lr_from_k(k):
    return rounded_lr(LR_SEARCH_BASE * LR_SEARCH_FACTOR**k)


def nearest_lr_k(lr):
    if lr <= 0:
        raise ValueError(f"LR must be positive, got {lr}")
    return int(round(log(lr / LR_SEARCH_BASE) / log(LR_SEARCH_FACTOR)))


def momentum_from_index(momentum_index):
    return MOMENTUM_SEARCH_VALUES[momentum_index]


def nearest_momentum_index(momentum):
    for index, search_momentum in enumerate(MOMENTUM_SEARCH_VALUES):
        if isclose(momentum, search_momentum, rel_tol=0.0, abs_tol=1e-12):
            return index
    raise ValueError(
        "momentum must be in %s, got %s" % (MOMENTUM_SEARCH_VALUES, momentum)
    )


def make_first_train_batch(batch_size):
    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size)
    train_loader.shuffle = False
    inputs, labels = next(iter(train_loader))
    train_images = train_loader.normalized_images()[:5000]
    return inputs.detach(), labels.detach(), train_images


def first_batch_loss(model, inputs, labels):
    model.train()
    with torch.inference_mode():
        outputs = model(inputs)
        loss = F.cross_entropy(
            outputs.float(),
            labels,
            label_smoothing=LABEL_SMOOTHING,
            reduction="mean",
        )
    return loss.item()


def make_optimizers(
    model,
    batch_size,
    muon_lr,
    muon_momentum,
    muon_nesterov,
    sgd_lr_mult,
    muon_orthogonalize,
):
    bias_lr = 104 * sgd_lr_mult
    head_lr = 1340 * sgd_lr_mult
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=0),
        dict(params=norm_biases, lr=bias_lr, weight_decay=0),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=0),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    optimizer2 = Muon(
        filter_params,
        lr=muon_lr,
        momentum=muon_momentum,
        nesterov=muon_nesterov,
        orthogonalize=muon_orthogonalize,
    )
    return optimizer1, optimizer2


def set_muon_hparams(muon_optimizer, muon_lr, muon_momentum, muon_nesterov):
    muon_lr = rounded_lr(muon_lr)
    for group in muon_optimizer.param_groups:
        group["lr"] = muon_lr
        group["momentum"] = muon_momentum
        group["nesterov"] = muon_nesterov
    return muon_lr


def snapshot_training_state(model, optimizers):
    return dict(
        model={
            name: value.detach().clone() for name, value in model.state_dict().items()
        },
        optimizers=[copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers],
    )


def load_training_state(model, optimizers, state):
    model.load_state_dict(state["model"])
    for optimizer, optimizer_state in zip(optimizers, state["optimizers"]):
        optimizer.load_state_dict(optimizer_state)
    model.zero_grad(set_to_none=True)


def train_one_step(
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    muon_lr,
    muon_momentum,
    muon_nesterov,
):
    set_muon_hparams(muon_optimizer, muon_lr, muon_momentum, muon_nesterov)
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(inputs, whiten_bias_grad=True)
    loss = F.cross_entropy(
        outputs,
        labels,
        label_smoothing=LABEL_SMOOTHING,
        reduction="mean",
    )
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    model.zero_grad(set_to_none=True)


def train_interval(
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    muon_lr,
    muon_momentum,
    muon_nesterov,
    interval_steps,
):
    muon_lr = rounded_lr(muon_lr)
    losses = [first_batch_loss(model, inputs, labels)]
    completed_steps = 0
    for _ in range(interval_steps):
        train_one_step(
            model,
            optimizers,
            muon_optimizer,
            inputs,
            labels,
            muon_lr,
            muon_momentum,
            muon_nesterov,
        )
        step_loss = first_batch_loss(model, inputs, labels)
        losses.append(step_loss)
        completed_steps += 1
        if not torch.isfinite(torch.tensor(step_loss)):
            break
    while len(losses) <= interval_steps:
        losses.append(float("inf"))
    return dict(
        muon_lr=muon_lr,
        muon_momentum=muon_momentum,
        muon_nesterov=muon_nesterov,
        losses=losses,
        completed_steps=completed_steps,
        final_loss=losses[interval_steps],
        end_state=snapshot_training_state(model, optimizers),
    )


def point_sort_key(point):
    k, momentum_index = point
    return (abs(k), k, momentum_index)


def better_point(point, incumbent_point, results_by_point):
    if incumbent_point is None:
        return True
    loss = results_by_point[point]["final_loss"]
    incumbent_loss = results_by_point[incumbent_point]["final_loss"]
    return (loss, point_sort_key(point)) < (
        incumbent_loss,
        point_sort_key(incumbent_point),
    )


def neighbor_points(point, lr_step, search_momentum):
    k, momentum_index = point
    yield (k - lr_step, momentum_index)
    yield (k + lr_step, momentum_index)
    if search_momentum:
        if momentum_index > 0:
            yield (k, momentum_index - 1)
        if momentum_index + 1 < len(MOMENTUM_SEARCH_VALUES):
            yield (k, momentum_index + 1)


def best_neighbor_point(middle_point, results_by_point, lr_step, search_momentum):
    best_point = middle_point
    for point in neighbor_points(middle_point, lr_step, search_momentum):
        if better_point(point, best_point, results_by_point):
            best_point = point
    return best_point


def find_best_lr_momentum_point(
    initial_lr_k,
    initial_momentum_index,
    evaluate,
    results_by_point,
    search_momentum,
):
    middle_point = (initial_lr_k, initial_momentum_index)
    initial_points = [middle_point]
    evaluate(middle_point)
    for point in neighbor_points(middle_point, 1, search_momentum):
        evaluate(point)
        initial_points.append(point)

    for point in initial_points:
        if better_point(point, middle_point, results_by_point):
            middle_point = point

    for _ in range(LR_SEARCH_MAX_MOVES):
        for point in neighbor_points(middle_point, 1, search_momentum):
            evaluate(point)
        next_point = best_neighbor_point(
            middle_point, results_by_point, 1, search_momentum
        )
        if next_point == middle_point:
            break
        middle_point = next_point
    else:
        raise RuntimeError(
            "LR/momentum search did not converge within %d moves" % LR_SEARCH_MAX_MOVES
        )

    return middle_point


def search_cooldown_lr(
    search_name,
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    initial_lr_k,
    initial_momentum_index,
    muon_nesterov,
    cooldown_steps,
    start_state,
    search_momentum,
):
    results_by_point = {}
    initial_lr = lr_from_k(initial_lr_k)
    initial_momentum = momentum_from_index(initial_momentum_index)

    def evaluate(point):
        k, momentum_index = point
        if point not in results_by_point:
            lr = lr_from_k(k)
            momentum = momentum_from_index(momentum_index)
            load_training_state(model, optimizers, start_state)
            result = train_interval(
                model=model,
                optimizers=optimizers,
                muon_optimizer=muon_optimizer,
                inputs=inputs,
                labels=labels,
                muon_lr=lr,
                muon_momentum=momentum,
                muon_nesterov=muon_nesterov,
                interval_steps=cooldown_steps,
            )
            result["k"] = k
            result["momentum_index"] = momentum_index
            result["initial_lr_k"] = initial_lr_k
            result["initial_lr"] = initial_lr
            result["initial_momentum_index"] = initial_momentum_index
            result["initial_momentum"] = initial_momentum
            results_by_point[point] = result
        return results_by_point[point]

    best_point = find_best_lr_momentum_point(
        initial_lr_k=initial_lr_k,
        initial_momentum_index=initial_momentum_index,
        evaluate=evaluate,
        results_by_point=results_by_point,
        search_momentum=search_momentum,
    )
    best_k, best_momentum_index = best_point
    best_result = results_by_point[best_point]
    load_training_state(model, optimizers, start_state)
    return dict(
        search_name=search_name,
        cooldown_steps=cooldown_steps,
        search_momentum=search_momentum,
        best_k=best_k,
        best_momentum_index=best_momentum_index,
        muon_lr=best_result["muon_lr"],
        muon_momentum=best_result["muon_momentum"],
        muon_nesterov=best_result["muon_nesterov"],
        initial_lr_k=initial_lr_k,
        initial_lr=initial_lr,
        initial_momentum_index=initial_momentum_index,
        initial_momentum=initial_momentum,
        initial_train_loss=best_result["losses"][0],
        final_train_loss=best_result["final_loss"],
        completed_steps=best_result["completed_steps"],
        losses=list(best_result["losses"]),
        search_evaluations=[
            dict(
                k=k,
                momentum_index=momentum_index,
                muon_lr=result["muon_lr"],
                muon_momentum=result["muon_momentum"],
                muon_nesterov=result["muon_nesterov"],
                initial_lr_k=result["initial_lr_k"],
                initial_lr=result["initial_lr"],
                initial_momentum_index=result["initial_momentum_index"],
                initial_momentum=result["initial_momentum"],
                losses=list(result["losses"]),
                final_loss=result["final_loss"],
                completed_steps=result["completed_steps"],
            )
            for (k, momentum_index), result in sorted(
                results_by_point.items()
            )
        ],
    )


def search_interval_lr(
    run,
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    initial_lr_k,
    initial_momentum_index,
    interval_steps,
    cooldown_steps,
    total_steps,
    interval_index,
    interval_start_step,
    batch_size,
    n_steps,
    search_momentum,
    muon_nesterov,
):
    search_name = "run%d_bs%d_N%d_M%d_interval%d_step%d" % (
        run,
        batch_size,
        n_steps,
        cooldown_steps,
        interval_index,
        interval_start_step,
    )
    start_state = snapshot_training_state(model, optimizers)
    results_by_point = {}
    initial_lr = lr_from_k(initial_lr_k)
    initial_momentum = momentum_from_index(initial_momentum_index)
    use_cooldown = interval_start_step + interval_steps + cooldown_steps <= total_steps
    use_cooldown = use_cooldown and cooldown_steps > 0

    def evaluate(point):
        k, momentum_index = point
        if point not in results_by_point:
            lr = lr_from_k(k)
            momentum = momentum_from_index(momentum_index)
            load_training_state(model, optimizers, start_state)
            result = train_interval(
                model=model,
                optimizers=optimizers,
                muon_optimizer=muon_optimizer,
                inputs=inputs,
                labels=labels,
                muon_lr=lr,
                muon_momentum=momentum,
                muon_nesterov=muon_nesterov,
                interval_steps=interval_steps,
            )
            interval_end_state = result["end_state"]
            result["interval_final_loss"] = result["final_loss"]
            result["cooldown_result"] = None
            if (
                use_cooldown
                and result["completed_steps"] == interval_steps
                and torch.isfinite(torch.tensor(result["final_loss"]))
            ):
                cooldown_search_name = "%s_cooldown_for_k%s_m%s_n%s" % (
                    search_name,
                    format_k(k),
                    format_momentum(momentum),
                    muon_nesterov,
                )
                cooldown_result = search_cooldown_lr(
                    search_name=cooldown_search_name,
                    model=model,
                    optimizers=optimizers,
                    muon_optimizer=muon_optimizer,
                    inputs=inputs,
                    labels=labels,
                    initial_lr_k=k,
                    initial_momentum_index=momentum_index,
                    muon_nesterov=muon_nesterov,
                    cooldown_steps=cooldown_steps,
                    start_state=interval_end_state,
                    search_momentum=False,
                )
                result["cooldown_result"] = cooldown_result
                result["final_loss"] = cooldown_result["final_train_loss"]
            result["k"] = k
            result["momentum_index"] = momentum_index
            result["initial_lr_k"] = initial_lr_k
            result["initial_lr"] = initial_lr
            result["initial_momentum_index"] = initial_momentum_index
            result["initial_momentum"] = initial_momentum
            results_by_point[point] = result
        return results_by_point[point]

    best_point = find_best_lr_momentum_point(
        initial_lr_k=initial_lr_k,
        initial_momentum_index=initial_momentum_index,
        evaluate=evaluate,
        results_by_point=results_by_point,
        search_momentum=search_momentum,
    )
    best_k, best_momentum_index = best_point
    best_result = results_by_point[best_point]
    load_training_state(model, optimizers, best_result["end_state"])
    best_cooldown_result = best_result["cooldown_result"]
    log_interval_lr_landscape(results_by_point)
    log_interval_lr_search_complete(
        best_result,
        best_cooldown_result,
        results_by_point,
    )
    return dict(
        search_name=search_name,
        interval_index=interval_index,
        interval_start_step=interval_start_step,
        interval_steps=interval_steps,
        cooldown_steps=cooldown_steps if use_cooldown else 0,
        search_momentum=search_momentum,
        best_k=best_k,
        best_momentum_index=best_momentum_index,
        muon_lr=best_result["muon_lr"],
        muon_momentum=best_result["muon_momentum"],
        muon_nesterov=best_result["muon_nesterov"],
        cooldown_best_k=(
            best_cooldown_result["best_k"] if best_cooldown_result else None
        ),
        cooldown_best_momentum_index=(
            best_cooldown_result["best_momentum_index"]
            if best_cooldown_result
            else None
        ),
        cooldown_muon_lr=(
            best_cooldown_result["muon_lr"] if best_cooldown_result else None
        ),
        cooldown_muon_momentum=(
            best_cooldown_result["muon_momentum"] if best_cooldown_result else None
        ),
        cooldown_muon_nesterov=(
            best_cooldown_result["muon_nesterov"] if best_cooldown_result else None
        ),
        initial_lr_k=initial_lr_k,
        initial_lr=initial_lr,
        initial_momentum_index=initial_momentum_index,
        initial_momentum=initial_momentum,
        cooldown_initial_lr_k=best_k,
        cooldown_initial_lr=lr_from_k(best_k),
        cooldown_initial_momentum_index=best_momentum_index,
        cooldown_initial_momentum=momentum_from_index(best_momentum_index),
        initial_train_loss=best_result["losses"][0],
        interval_final_train_loss=best_result["interval_final_loss"],
        final_train_loss=best_result["final_loss"],
        completed_steps=best_result["completed_steps"],
        losses=list(best_result["losses"]),
        cooldown_result=best_cooldown_result,
        search_evaluations=[
            dict(
                k=k,
                momentum_index=momentum_index,
                muon_lr=result["muon_lr"],
                muon_momentum=result["muon_momentum"],
                muon_nesterov=result["muon_nesterov"],
                initial_lr_k=result["initial_lr_k"],
                initial_lr=result["initial_lr"],
                initial_momentum_index=result["initial_momentum_index"],
                initial_momentum=result["initial_momentum"],
                interval_final_loss=result["interval_final_loss"],
                losses=list(result["losses"]),
                final_loss=result["final_loss"],
                completed_steps=result["completed_steps"],
                cooldown_result=result["cooldown_result"],
            )
            for (k, momentum_index), result in sorted(
                results_by_point.items()
            )
        ],
    )


def run_overfit_n_search(
    run,
    model,
    batch_size,
    n_steps,
    m_steps,
    muon_orthogonalize,
    initial_lr,
    initial_momentum,
    search_momentum,
    muon_nesterov,
    momentum_config_name,
    sgd_lr_mult,
):
    set_training_seed()
    initial_lr_k = nearest_lr_k(initial_lr)
    initial_lr = lr_from_k(initial_lr_k)
    initial_momentum_index = nearest_momentum_index(initial_momentum)
    initial_momentum = momentum_from_index(initial_momentum_index)
    inputs, labels, train_images = make_first_train_batch(batch_size)
    model.reset()
    model.init_whiten(train_images)
    optimizer1, optimizer2 = make_optimizers(
        model,
        batch_size=batch_size,
        muon_lr=rounded_lr(initial_lr),
        muon_momentum=initial_momentum,
        muon_nesterov=muon_nesterov,
        sgd_lr_mult=sgd_lr_mult,
        muon_orthogonalize=muon_orthogonalize,
    )
    optimizers = [optimizer1, optimizer2]

    losses = [first_batch_loss(model, inputs, labels)]
    log_train_loss(
        run=run,
        step=0,
        total_steps=OVERFIT_TRAIN_STEPS,
        loss=losses[-1],
        head_lr=optimizer1.param_groups[2]["lr"],
        muon_lr=optimizer2.param_groups[0]["lr"],
        muon_momentum=optimizer2.param_groups[0]["momentum"],
        muon_nesterov=optimizer2.param_groups[0]["nesterov"],
    )

    interval_results = []
    interval_initial_lr_k = initial_lr_k
    interval_initial_momentum_index = initial_momentum_index
    completed_steps = 0
    interval_index = 0
    while completed_steps < OVERFIT_TRAIN_STEPS:
        interval_steps = min(n_steps, OVERFIT_TRAIN_STEPS - completed_steps)
        interval_result = search_interval_lr(
            run=run,
            model=model,
            optimizers=optimizers,
            muon_optimizer=optimizer2,
            inputs=inputs,
            labels=labels,
            initial_lr_k=interval_initial_lr_k,
            initial_momentum_index=interval_initial_momentum_index,
            interval_steps=interval_steps,
            cooldown_steps=m_steps,
            total_steps=OVERFIT_TRAIN_STEPS,
            interval_index=interval_index,
            interval_start_step=completed_steps,
            batch_size=batch_size,
            n_steps=n_steps,
            search_momentum=search_momentum,
            muon_nesterov=muon_nesterov,
        )
        interval_results.append(interval_result)
        interval_initial_lr_k = interval_result["best_k"]
        interval_initial_momentum_index = interval_result["best_momentum_index"]
        actual_losses = interval_result["losses"][
            1 : 1 + interval_result["completed_steps"]
        ]
        for local_offset, loss in enumerate(actual_losses, start=1):
            global_step = completed_steps + local_offset
            losses.append(loss)
            log_train_loss(
                run=run,
                step=global_step,
                total_steps=OVERFIT_TRAIN_STEPS,
                loss=loss,
                head_lr=optimizer1.param_groups[2]["lr"],
                muon_lr=interval_result["muon_lr"],
                muon_momentum=interval_result["muon_momentum"],
                muon_nesterov=interval_result["muon_nesterov"],
            )
        completed_steps += interval_result["completed_steps"]
        interval_index += 1
        if not torch.isfinite(torch.tensor(losses[-1])):
            break

    while len(losses) <= OVERFIT_TRAIN_STEPS:
        losses.append(float("inf"))

    last_cooldown_result = next(
        (
            interval_result
            for interval_result in reversed(interval_results)
            if interval_result["cooldown_best_k"] is not None
        ),
        None,
    )

    return dict(
        run=run,
        batch_size=batch_size,
        n_steps=n_steps,
        m_steps=m_steps,
        muon_orthogonalize=muon_orthogonalize,
        momentum_config_name=momentum_config_name,
        search_momentum=search_momentum,
        muon_nesterov=muon_nesterov,
        initial_lr_k=initial_lr_k,
        initial_lr=initial_lr,
        initial_momentum_index=initial_momentum_index,
        initial_momentum=initial_momentum,
        final_muon_lr=interval_results[-1]["muon_lr"] if interval_results else None,
        final_muon_lr_k=interval_results[-1]["best_k"] if interval_results else None,
        final_muon_momentum=(
            interval_results[-1]["muon_momentum"] if interval_results else None
        ),
        final_muon_nesterov=(
            interval_results[-1]["muon_nesterov"] if interval_results else None
        ),
        final_muon_momentum_index=(
            interval_results[-1]["best_momentum_index"] if interval_results else None
        ),
        final_cooldown_muon_lr=(
            last_cooldown_result["cooldown_muon_lr"] if last_cooldown_result else None
        ),
        final_cooldown_muon_momentum=(
            last_cooldown_result["cooldown_muon_momentum"]
            if last_cooldown_result
            else None
        ),
        final_cooldown_muon_nesterov=(
            last_cooldown_result["cooldown_muon_nesterov"]
            if last_cooldown_result
            else None
        ),
        final_cooldown_muon_lr_k=(
            last_cooldown_result["cooldown_best_k"] if last_cooldown_result else None
        ),
        final_cooldown_muon_momentum_index=(
            last_cooldown_result["cooldown_best_momentum_index"]
            if last_cooldown_result
            else None
        ),
        sgd_lr_mult=sgd_lr_mult,
        losses=losses,
        initial_train_loss=losses[0],
        final_train_loss=losses[OVERFIT_TRAIN_STEPS],
        steps=completed_steps,
        target_steps=OVERFIT_TRAIN_STEPS,
        interval_results=interval_results,
    )


def main():
    # We re-use the model object between runs to save non-data-dependent setup time.
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    results = []
    for config in RUN_CONFIGS:
        for n_steps in N_SEARCH_STEPS:
            for m_steps in M_COOLDOWN_STEPS:
                for muon_orthogonalize in MUON_ORTHOGONALIZE:
                    for momentum_config in MUON_MOMENTUM_CONFIGS:
                        initial_momentum = momentum_config["initial_momentum"]
                        run = len(results)
                        print(
                            "cifar_baseline2_overfit_n_search run=%d batch_size=%d "
                            "N=%d M=%d muon_orthogonalize=%s "
                            "momentum_config=%s search_momentum=%s "
                            "muon_nesterov=%s "
                            "initial_muon_lr=%.6g initial_muon_lr_k=%d "
                            "initial_muon_momentum=%s "
                            "initial_muon_momentum_index=%d"
                            % (
                                run,
                                config["batch_size"],
                                n_steps,
                                m_steps,
                                muon_orthogonalize,
                                momentum_config["momentum_config_name"],
                                momentum_config["search_momentum"],
                                momentum_config["muon_nesterov"],
                                config["initial_lr"],
                                nearest_lr_k(config["initial_lr"]),
                                format_momentum(initial_momentum),
                                nearest_momentum_index(initial_momentum),
                            ),
                            flush=True,
                        )
                        run_start_time = time.perf_counter()
                        result = run_overfit_n_search(
                            run=run,
                            model=model,
                            batch_size=config["batch_size"],
                            n_steps=n_steps,
                            m_steps=m_steps,
                            muon_orthogonalize=muon_orthogonalize,
                            initial_lr=config["initial_lr"],
                            initial_momentum=initial_momentum,
                            search_momentum=momentum_config["search_momentum"],
                            muon_nesterov=momentum_config["muon_nesterov"],
                            momentum_config_name=momentum_config[
                                "momentum_config_name"
                            ],
                            sgd_lr_mult=config["sgd_lr_mult"],
                        )
                        result["run_seconds"] = time.perf_counter() - run_start_time
                        results.append(result)
                        print("Batch size:          %d" % result["batch_size"])
                        print("N steps:             %d" % result["n_steps"])
                        print("M cooldown steps:    %d" % result["m_steps"])
                        print("Muon orthogonalize:  %s" % result["muon_orthogonalize"])
                        print(
                            "Momentum config:     %s" % result["momentum_config_name"]
                        )
                        print("Search momentum:     %s" % result["search_momentum"])
                        print("Muon nesterov:       %s" % result["muon_nesterov"])
                        print("Initial Muon lr:     %.6g" % result["initial_lr"])
                        print("Initial Muon lr k:   %d" % result["initial_lr_k"])
                        print(
                            "Initial Muon mom:    %s"
                            % format_momentum(result["initial_momentum"])
                        )
                        print(
                            "Initial Muon mom i:  %d"
                            % result["initial_momentum_index"]
                        )
                        print("Final Muon lr:       %.6g" % result["final_muon_lr"])
                        print(
                            "Final Muon lr k:     %s"
                            % format_k(result["final_muon_lr_k"])
                        )
                        print(
                            "Final Muon momentum: %s"
                            % format_momentum(result["final_muon_momentum"])
                        )
                        print(
                            "Final Muon nesterov: %s" % result["final_muon_nesterov"]
                        )
                        print(
                            "Final Muon mom i:    %s"
                            % format_k(result["final_muon_momentum_index"])
                        )
                        if result["final_cooldown_muon_lr"] is not None:
                            print(
                                "Final cooldown lr:   %.6g"
                                % result["final_cooldown_muon_lr"]
                            )
                            print(
                                "Final cooldown lr k: %s"
                                % format_k(result["final_cooldown_muon_lr_k"])
                            )
                            print(
                                "Final cooldown mom:  %s"
                                % format_momentum(
                                    result["final_cooldown_muon_momentum"]
                                )
                            )
                            print(
                                "Final cooldown nest: %s"
                                % result["final_cooldown_muon_nesterov"]
                            )
                            print(
                                "Final cooldown mom i: %s"
                                % format_k(
                                    result["final_cooldown_muon_momentum_index"]
                                )
                            )
                        print("SGD lr mult:         %.6g" % result["sgd_lr_mult"])
                        print(
                            "Initial train loss:  %.6f" % result["initial_train_loss"]
                        )
                        print(
                            "Final train loss:    %.6f" % result["final_train_loss"]
                        )
                        print("Run seconds:         %.3f" % result["run_seconds"])

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, results=results), log_path)
    print(os.path.abspath(log_path), flush=True)


if __name__ == "__main__":
    main()
