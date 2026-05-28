"""
airbench94_muon.py
Runs in 2.59 seconds on a 400W NVIDIA A100 using torch==2.4.1
Attains 94.01 mean accuracy (n=200 trials)
Descends from https://github.com/tysam-code/hlb-CIFAR10/blob/main/main.py
"""

#############################################
#                  Setup                    #
#############################################

import os
import sys
import time

with open(sys.argv[0]) as f:
    code = f.read()
from collections import Counter
import json
import uuid
from math import ceil, floor, isfinite, log10

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
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
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


def log_training_batch_losses(step, total_steps, update, losses):
    print(
        "training_batch_losses "
        f"step={step}/{total_steps} update={update} losses={json.dumps(losses)}",
        flush=True,
    )


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


def log_eval(run, epoch, val_acc, time_seconds):
    run_info = f" run={run}" if run is not None else ""
    print(
        f"eval{run_info} epoch={epoch} val_acc={val_acc:.4f} "
        f"time_seconds={time_seconds:.4f}",
        flush=True,
    )


def log_final_eval(
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    tta_val_acc,
    time_seconds,
):
    print(
        f"eval epoch=final train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} train_acc={train_acc:.4f} "
        f"val_acc={val_acc:.4f} tta_val_acc={tta_val_acc:.4f} "
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


############################################
#               Evaluation                 #
############################################


def infer(model, loader, tta_level=0):

    # Test-time augmentation strategy (for tta_level=2):
    # 1. Flip/mirror the image left-to-right (50% of the time).
    # 2. Translate the image by one pixel either up-and-left or down-and-right (50% of the time,
    #    i.e. both happen 25% of the time).
    #
    # This creates 6 views per image (left/right times the two translations and no-translation),
    # which we evaluate and then weight according to the given probabilities.

    def infer_basic(inputs, net):
        return net(inputs).clone()

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

    model.eval()
    test_images = loader.normalized_images()
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.inference_mode():
        return torch.cat(
            [infer_fn(inputs, model) for inputs in test_images.split(2000)]
        )


def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()


def evaluate_loader_loss_and_accuracy(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    images = loader.normalized_images()
    with torch.inference_mode():
        for inputs, labels in zip(images.split(2000), loader.labels.split(2000)):
            outputs = model(inputs)
            total_loss += F.cross_entropy(
                outputs.float(), labels, label_smoothing=0.2, reduction="sum"
            ).item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_examples += len(labels)
    return total_loss / total_examples, total_correct / total_examples


def batchnorm_modules(model):
    return [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]


def snapshot_batchnorm_buffers(model):
    return [
        (
            module,
            None
            if module.running_mean is None
            else module.running_mean.detach().clone(),
            None if module.running_var is None else module.running_var.detach().clone(),
            None
            if module.num_batches_tracked is None
            else module.num_batches_tracked.detach().clone(),
        )
        for module in batchnorm_modules(model)
    ]


def restore_batchnorm_buffers(states):
    for module, running_mean, running_var, num_batches_tracked in states:
        if running_mean is not None:
            module.running_mean.copy_(running_mean)
        if running_var is not None:
            module.running_var.copy_(running_var)
        if num_batches_tracked is not None:
            module.num_batches_tracked.copy_(num_batches_tracked)


def evaluate_training_batch_losses(model, batches):
    was_training = model.training
    batchnorm_states = snapshot_batchnorm_buffers(model)
    model.eval()
    for module in batchnorm_modules(model):
        module.train()

    losses = []
    try:
        with torch.no_grad():
            for inputs, labels in batches:
                outputs = model(inputs)
                loss = F.cross_entropy(
                    outputs.float(), labels, label_smoothing=0.2, reduction="mean"
                )
                losses.append(loss.item())
    finally:
        restore_batchnorm_buffers(batchnorm_states)
        model.train(was_training)

    return losses


def evaluate_training_loss_and_accuracy(model, batches):
    was_training = model.training
    batchnorm_states = snapshot_batchnorm_buffers(model)
    model.eval()
    for module in batchnorm_modules(model):
        module.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    try:
        with torch.no_grad():
            for inputs, labels in batches:
                outputs = model(inputs)
                total_loss += F.cross_entropy(
                    outputs.float(), labels, label_smoothing=0.2, reduction="sum"
                ).item()
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_examples += len(labels)
    finally:
        restore_batchnorm_buffers(batchnorm_states)
        model.train(was_training)

    return total_loss / total_examples, total_correct / total_examples


def materialize_training_batches(train_loader, total_steps):
    batches = []
    while len(batches) < total_steps:
        for inputs, labels in train_loader:
            batches.append((inputs.detach(), labels.detach()))
            if len(batches) >= total_steps:
                break
    return batches


def materialize_first_training_batch(train_loader, total_steps):
    inputs, labels = next(iter(train_loader))
    batch = (inputs.detach(), labels.detach())
    return [batch for _ in range(total_steps)]


def every_training_step(total_steps):
    return list(range(1, total_steps + 1))


def epoch_end_steps(total_steps, steps_per_epoch):
    steps = list(range(steps_per_epoch, total_steps + 1, steps_per_epoch))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


############################################
#              Line Search                 #
############################################

BEST_LR_FACTOR = 0.8
BEST_LR_EMA_MOMENTUM = 0.5
BEST_LR_MAX_SEARCH_STEPS = 40
BEST_LR_REL_DIFF_THRESHOLD = 0.4
END_LR_MULTIPLIER = 0.01
PEAK_LR_MAX_LEFT_STEPS = 30
PEAK_LR_MAX_RIGHT_STEPS = 20
PEAK_LR_HISTOGRAM_RADIUS = 5
PEAK_LR_LOCAL_NEIGHBOR_DISTANCE = 2
PEAK_LR_DISCARDED_EDGE_STEPS = 1
PEAK_LR_INITIAL_SIDE_STEPS = (
    PEAK_LR_HISTOGRAM_RADIUS
    + PEAK_LR_LOCAL_NEIGHBOR_DISTANCE
    + PEAK_LR_DISCARDED_EDGE_STEPS
)
LOG_PRE_POST_LOSSES = False
LOG_PRE_UPDATE_LOSSES = False
LOG_POST_UPDATE_LOSSES = False
LOG_BEST_LR = True


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
        group["lr"] = lr


def finite_loss_item(loss):
    return loss.item() if torch.isfinite(loss) else float("inf")


def format_lr_loss_map(losses_by_lr):
    return {"%.8g" % lr: loss for lr, loss in sorted(losses_by_lr.items())}


def evaluate_candidate_lr(
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    whiten_bias_grad,
    lr,
    search_state,
    losses_by_lr,
):
    if lr in losses_by_lr:
        restore_training_state(model, optimizers, search_state)
        return losses_by_lr[lr]

    restore_training_state(model, optimizers, search_state)
    set_muon_lr(muon_optimizer, lr)
    for opt in optimizers:
        opt.step()
    with torch.no_grad():
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        loss = F.cross_entropy(
            outputs.float(), labels, label_smoothing=0.2, reduction="mean"
        )
    losses_by_lr[lr] = finite_loss_item(loss)
    restore_training_state(model, optimizers, search_state)
    return losses_by_lr[lr]


def choose_best_lr(
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    whiten_bias_grad,
    init_lr,
):
    search_state = capture_training_state(model, optimizers)
    losses_by_lr = {}

    def candidate_loss(lr):
        return evaluate_candidate_lr(
            model,
            optimizers,
            muon_optimizer,
            inputs,
            labels,
            whiten_bias_grad,
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
            restore_training_state(model, optimizers, search_state)
            return center_lr, center_loss, losses_by_lr
        center_lr = left_lr if left_loss < right_loss else right_lr

    best_lr, best_loss = min(losses_by_lr.items(), key=lambda item: item[1])
    restore_training_state(model, optimizers, search_state)
    return best_lr, best_loss, losses_by_lr


def relative_loss_diff(loss, best_loss, zero_loss):
    denominator = zero_loss - best_loss
    if denominator <= 0:
        return float("inf")
    return (loss - best_loss) / denominator


def largest_lr_within_rel_diff(losses_by_lr, rel_diff_threshold):
    finite_losses = {
        lr: loss for lr, loss in losses_by_lr.items() if lr >= 0 and isfinite(loss)
    }
    if not finite_losses:
        return 0.0, float("inf"), float("inf")

    best_loss = min(finite_losses.values())
    zero_loss = finite_losses.get(0.0, float("inf"))
    candidates = [
        (lr, loss)
        for lr, loss in finite_losses.items()
        if relative_loss_diff(loss, best_loss, zero_loss) <= rel_diff_threshold
    ]
    if not candidates:
        best_lr, selected_loss = min(finite_losses.items(), key=lambda item: item[1])
        return (
            best_lr,
            selected_loss,
            relative_loss_diff(selected_loss, best_loss, zero_loss),
        )

    selected_lr, selected_loss = max(candidates, key=lambda item: item[0])
    return (
        selected_lr,
        selected_loss,
        relative_loss_diff(selected_loss, best_loss, zero_loss),
    )


def choose_largest_rel_loss_lr(
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    whiten_bias_grad,
    init_lr,
    rel_diff_threshold,
):
    best_lr, best_loss, losses_by_lr = choose_best_lr(
        model,
        optimizers,
        muon_optimizer,
        inputs,
        labels,
        whiten_bias_grad,
        init_lr,
    )
    search_state = capture_training_state(model, optimizers)

    def candidate_loss(lr):
        return evaluate_candidate_lr(
            model,
            optimizers,
            muon_optimizer,
            inputs,
            labels,
            whiten_bias_grad,
            lr,
            search_state,
            losses_by_lr,
        )

    candidate_loss(0.0)
    center_lr = max(best_lr, init_lr)
    candidate_loss(center_lr)
    for _ in range(BEST_LR_MAX_SEARCH_STEPS):
        selected_lr, _, selected_rel_diff = largest_lr_within_rel_diff(
            losses_by_lr, rel_diff_threshold
        )
        largest_lr = max(lr for lr in losses_by_lr if lr >= 0)
        if (
            selected_lr < largest_lr
            or not isfinite(selected_rel_diff)
            or selected_rel_diff > rel_diff_threshold
        ):
            break
        candidate_loss(largest_lr / BEST_LR_FACTOR)

    restore_training_state(model, optimizers, search_state)
    selected_lr, selected_loss, _ = largest_lr_within_rel_diff(
        losses_by_lr, rel_diff_threshold
    )
    return selected_lr, selected_loss, losses_by_lr


def lr_grid_around(base_lr, left_steps, right_steps):
    return [
        0.0,
        *[
            float("%.8g" % (base_lr * (BEST_LR_FACTOR**exponent)))
            for exponent in range(left_steps, -right_steps - 1, -1)
        ],
    ]


def evaluate_candidate_lr_sample_losses(
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    whiten_bias_grad,
    lr,
    search_state,
    sample_losses_by_lr,
):
    if lr in sample_losses_by_lr:
        restore_training_state(model, optimizers, search_state)
        return sample_losses_by_lr[lr]

    restore_training_state(model, optimizers, search_state)
    set_muon_lr(muon_optimizer, lr)
    for opt in optimizers:
        opt.step()
    with torch.no_grad():
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        losses = F.cross_entropy(
            outputs.float(), labels, label_smoothing=0.2, reduction="none"
        )
    sample_losses_by_lr[lr] = losses.detach()
    restore_training_state(model, optimizers, search_state)
    return sample_losses_by_lr[lr]


def peak_lr_valid_positions(lr_grid):
    return range(2, len(lr_grid) - 1)


def peak_lr_candidate_positions(lr_grid):
    margin = PEAK_LR_HISTOGRAM_RADIUS + PEAK_LR_LOCAL_NEIGHBOR_DISTANCE
    valid_positions = peak_lr_valid_positions(lr_grid)
    valid_start = valid_positions.start
    valid_stop = valid_positions.stop - 1
    return range(valid_start + margin, valid_stop - margin + 1)


def triangle_smoothed_peak_scores(counts_by_lr, lr_grid):
    valid_positions = set(peak_lr_valid_positions(lr_grid))
    counts = [int(counts_by_lr.get(lr, 0)) for lr in lr_grid]
    radius = PEAK_LR_HISTOGRAM_RADIUS
    return {
        position: sum(
            (radius + 1 - abs(offset)) * counts[position + offset]
            for offset in range(-radius, radius + 1)
            if position + offset in valid_positions
        )
        for position in valid_positions
    }


def local_smoothed_peak_position(counts_by_lr, lr_grid, init_position):
    scores = triangle_smoothed_peak_scores(counts_by_lr, lr_grid)
    candidate_positions = list(peak_lr_candidate_positions(lr_grid))
    if not candidate_positions:
        return None

    local_peaks = []
    for position in candidate_positions:
        score = scores[position]
        if score <= 0:
            continue
        if all(
            score > scores[position + offset]
            for offset in range(
                -PEAK_LR_LOCAL_NEIGHBOR_DISTANCE,
                PEAK_LR_LOCAL_NEIGHBOR_DISTANCE + 1,
            )
            if offset != 0
        ):
            local_peaks.append(position)
    if not local_peaks:
        return None

    return max(
        local_peaks,
        key=lambda position: (scores[position], -abs(position - init_position)),
    )


def fallback_smoothed_peak_position(counts_by_lr, lr_grid):
    scores = triangle_smoothed_peak_scores(counts_by_lr, lr_grid)
    candidate_positions = list(peak_lr_candidate_positions(lr_grid))
    if not candidate_positions:
        return None
    return max(candidate_positions, key=lambda position: scores[position])


def peak_lr_extension_sides(counts_by_lr, lr_grid):
    scores = triangle_smoothed_peak_scores(counts_by_lr, lr_grid)
    candidate_positions = list(peak_lr_candidate_positions(lr_grid))
    if not candidate_positions:
        return True, True

    position = max(candidate_positions, key=lambda candidate: scores[candidate])
    score = scores[position]
    blockers = [
        (position + offset, scores[position + offset])
        for offset in range(
            -PEAK_LR_LOCAL_NEIGHBOR_DISTANCE,
            PEAK_LR_LOCAL_NEIGHBOR_DISTANCE + 1,
        )
        if offset != 0 and scores[position + offset] >= score
    ]
    left_blockers = [item for item in blockers if item[0] < position]
    right_blockers = [item for item in blockers if item[0] > position]
    if left_blockers and right_blockers:
        left_score = max(score for _, score in left_blockers)
        right_score = max(score for _, score in right_blockers)
        if left_score == right_score:
            return True, True
        return left_score > right_score, right_score > left_score
    if left_blockers:
        return True, False
    if right_blockers:
        return False, True
    return True, True


def peak_lr_counts_for_grid(sample_losses_by_lr, lr_grid):
    sample_selection_lrs = lr_grid[2:-1]
    losses = torch.stack(
        [
            torch.where(
                torch.isfinite(sample_losses_by_lr[lr]),
                sample_losses_by_lr[lr],
                torch.full_like(sample_losses_by_lr[lr], float("inf")),
            )
            for lr in sample_selection_lrs
        ]
    )
    best_indices = losses.argmin(dim=0).detach().cpu().tolist()
    return Counter(sample_selection_lrs[index] for index in best_indices)


def choose_peak_lr(
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    whiten_bias_grad,
    init_lr,
):
    search_state = capture_training_state(model, optimizers)
    left_steps = PEAK_LR_INITIAL_SIDE_STEPS
    right_steps = PEAK_LR_INITIAL_SIDE_STEPS
    sample_losses_by_lr = {}
    selected_position = None

    while True:
        lr_grid = lr_grid_around(init_lr, left_steps, right_steps)
        for lr in lr_grid:
            evaluate_candidate_lr_sample_losses(
                model,
                optimizers,
                muon_optimizer,
                inputs,
                labels,
                whiten_bias_grad,
                lr,
                search_state,
                sample_losses_by_lr,
            )

        optimal_lr_counts = peak_lr_counts_for_grid(sample_losses_by_lr, lr_grid)
        selected_position = local_smoothed_peak_position(
            optimal_lr_counts, lr_grid, 1 + left_steps
        )
        if selected_position is not None:
            break

        can_extend_left = left_steps < PEAK_LR_MAX_LEFT_STEPS
        can_extend_right = right_steps < PEAK_LR_MAX_RIGHT_STEPS
        if not can_extend_left and not can_extend_right:
            selected_position = fallback_smoothed_peak_position(
                optimal_lr_counts, lr_grid
            )
            break
        extend_left, extend_right = peak_lr_extension_sides(optimal_lr_counts, lr_grid)
        if extend_left and not can_extend_left:
            extend_right = True
        if extend_right and not can_extend_right:
            extend_left = True
        if can_extend_left and extend_left:
            left_steps += 1
        if can_extend_right and extend_right:
            right_steps += 1

    peak_lr = (
        lr_grid[selected_position] if selected_position is not None else float("nan")
    )
    if not (isfinite(peak_lr) and peak_lr >= 0):
        peak_lr = init_lr

    losses_by_lr = {
        lr: finite_loss_item(sample_losses_by_lr[lr].mean()) for lr in lr_grid
    }
    peak_loss = losses_by_lr.get(peak_lr, float("inf"))
    restore_training_state(model, optimizers, search_state)
    return peak_lr, peak_loss, losses_by_lr


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


############################################
#                Training                  #
############################################

OVERFIT_BATCH_SIZES = [500, 2000, 5000, 10000]
OVERFIT_TRAIN_STEPS = 10
OVERFIT_MUON_LR = 0.19
OVERFIT_SGD_LR_MULT = 1.0
SCHEDULE_SEARCH_BASE_LR = 0.2
SCHEDULE_SEARCH_FACTOR = 0.8
SCHEDULE_SEARCH_INITIAL_EXPONENTS = (0, 2, 10)
SCHEDULE_SEARCH_INITIAL_POINT = SCHEDULE_SEARCH_INITIAL_EXPONENTS
SCHEDULE_SEARCH_MAX_MOVES = 30
APPLIED_LR_SCHEDULES = {}


def round_sigfigs(value, sigfigs=2):
    if value == 0:
        return 0.0
    return round(value, sigfigs - 1 - floor(log10(abs(value))))


def schedule_value(exponent):
    return SCHEDULE_SEARCH_BASE_LR * (SCHEDULE_SEARCH_FACTOR**exponent)


def schedule_cache_key(point):
    return tuple(schedule_value(exponent) for exponent in point)


def point_valid(point):
    return len(point) == 3


def schedule_from_point(point):
    step1_lr, step5_lr, step10_lr = schedule_cache_key(point)
    values = []
    for one_indexed_step in range(1, OVERFIT_TRAIN_STEPS + 1):
        if one_indexed_step <= 5:
            progress = (one_indexed_step - 1) / 4
            lr = step1_lr + (step5_lr - step1_lr) * progress
        else:
            progress = (one_indexed_step - 5) / 5
            lr = step5_lr + (step10_lr - step5_lr) * progress
        values.append(float("%.8g" % lr))
    values[-1] = step10_lr
    return tuple(values)


def lr_label(value):
    return ("%.8g" % value).replace("-", "m").replace(".", "p")


def point_label(point):
    step1_lr, step5_lr, step10_lr = schedule_cache_key(point)
    return (
        f"step1_k{point[0]}_{lr_label(step1_lr)}_"
        f"step5_k{point[1]}_{lr_label(step5_lr)}_"
        f"step10_k{point[2]}_{lr_label(step10_lr)}"
    )


def scheduler_name_from_point(point):
    return f"applied_lr_{point_label(point)}"


def register_schedule(point):
    scheduler = scheduler_name_from_point(point)
    if scheduler not in APPLIED_LR_SCHEDULES:
        APPLIED_LR_SCHEDULES[scheduler] = schedule_from_point(point)
    return scheduler


def overfit_run_config(batch_size, point):
    scheduler = register_schedule(point)
    return dict(
        name=f"overfit_bs{batch_size}_{scheduler}",
        batch_size=batch_size,
        muon_lr=OVERFIT_MUON_LR,
        sgd_lr_mult=OVERFIT_SGD_LR_MULT,
        best_lr_strategy=None,
        best_lr_scheduler=scheduler,
    )


def schedule_search_neighbors(point):
    neighbors = []
    for index in range(len(point)):
        for offset in (-1, 1):
            neighbor = list(point)
            neighbor[index] += offset
            neighbor = tuple(neighbor)
            if point_valid(neighbor):
                neighbors.append(neighbor)
    return neighbors


def main(
    run,
    model,
    name,
    batch_size,
    muon_lr,
    sgd_lr_mult=None,
    best_lr_strategy=None,
    best_lr_rel_diff_threshold=BEST_LR_REL_DIFF_THRESHOLD,
    best_lr_linear_decay=False,
    best_lr_scheduler=None,
):
    run_id = run
    run_wall_start = time.perf_counter()
    set_training_seed()
    use_best_lr = best_lr_strategy is not None
    best_lr_scheduler = resolve_best_lr_scheduler(
        best_lr_scheduler, best_lr_linear_decay
    )
    best_lr_linear_decay = uses_best_lr_decay(best_lr_scheduler)

    SGD_LR_MULT = batch_size / 2000 if sgd_lr_mult is None else sgd_lr_mult
    bias_lr = 104 * SGD_LR_MULT
    head_lr = 1340 * SGD_LR_MULT

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
    total_train_steps = OVERFIT_TRAIN_STEPS
    whiten_bias_train_steps = total_train_steps

    # Create optimizers and learning rate schedulers
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
    optimizer2 = Muon(filter_params, lr=muon_lr, momentum=0.6, nesterov=True)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # For accurately timing GPU code
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

    model.reset()
    step = 0

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalized_images()[:5000]
    model.init_whiten(train_images)
    stop_timer()

    training_batches = materialize_first_training_batch(
        train_loader, total_train_steps
    )
    loss_log_steps_list = [total_train_steps] if LOG_PRE_POST_LOSSES else []
    loss_log_steps = set(loss_log_steps_list)
    training_batch_loss_logs = []
    best_lr_ema = muon_lr
    best_lr_logs = []

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        epoch_start = epoch * len(train_loader)
        epoch_batches = training_batches[epoch_start : epoch_start + len(train_loader)]
        for inputs, labels in epoch_batches:
            whiten_bias_grad = step < whiten_bias_train_steps
            outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
            loss = F.cross_entropy(
                outputs, labels, label_smoothing=0.2, reduction="mean"
            )
            loss.backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:]:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            if best_lr_scheduler in APPLIED_LR_SCHEDULES:
                set_muon_lr(optimizer2, APPLIED_LR_SCHEDULES[best_lr_scheduler][step])
            elif not use_best_lr:
                for group in optimizer2.param_groups:
                    group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            if LOG_PRE_UPDATE_LOSSES and step + 1 in loss_log_steps:
                stop_timer()
                pre_update_losses = evaluate_training_batch_losses(
                    model, training_batches
                )
                training_batch_loss_logs.append(
                    dict(step=step + 1, update="pre", losses=pre_update_losses)
                )
                log_training_batch_losses(
                    step + 1, total_train_steps, "pre", pre_update_losses
                )
                start_timer()
            if use_best_lr:
                lr_decay_multiplier = best_lr_scheduler_multiplier(
                    step, total_train_steps, len(train_loader), best_lr_scheduler
                )
                init_lr = best_lr_ema
                if best_lr_strategy == "min_loss":
                    searched_lr, best_loss, losses_by_lr = choose_best_lr(
                        model,
                        optimizers,
                        optimizer2,
                        inputs,
                        labels,
                        whiten_bias_grad,
                        init_lr,
                    )
                elif best_lr_strategy == "largest_rel":
                    searched_lr, best_loss, losses_by_lr = choose_largest_rel_loss_lr(
                        model,
                        optimizers,
                        optimizer2,
                        inputs,
                        labels,
                        whiten_bias_grad,
                        init_lr,
                        best_lr_rel_diff_threshold,
                    )
                elif best_lr_strategy == "peak_lr":
                    searched_lr, best_loss, losses_by_lr = choose_peak_lr(
                        model,
                        optimizers,
                        optimizer2,
                        inputs,
                        labels,
                        whiten_bias_grad,
                        init_lr,
                    )
                else:
                    raise ValueError(f"Unknown best_lr_strategy: {best_lr_strategy}")
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
                set_muon_lr(optimizer2, actual_lr)
            applied_muon_lr = optimizer2.param_groups[0]["lr"]
            log_applied_lr(step + 1, total_train_steps, name, applied_muon_lr)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if LOG_POST_UPDATE_LOSSES and step in loss_log_steps:
                stop_timer()
                post_update_losses = evaluate_training_batch_losses(
                    model, training_batches
                )
                training_batch_loss_logs.append(
                    dict(step=step, update="post", losses=post_update_losses)
                )
                log_training_batch_losses(
                    step, total_train_steps, "post", post_update_losses
                )
                start_timer()
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        val_acc = evaluate(model, test_loader, tta_level=0)
        log_eval(run, epoch, val_acc, time_seconds)
        run = None  # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    train_loss, train_acc = evaluate_training_loss_and_accuracy(model, training_batches)
    val_loss, val_acc = evaluate_loader_loss_and_accuracy(model, test_loader)
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    log_final_eval(train_loss, val_loss, train_acc, val_acc, tta_val_acc, time_seconds)
    wall_time_seconds = time.perf_counter() - run_wall_start
    log_run_time(run_id, name, wall_time_seconds, time_seconds)

    return dict(
        train_loss=train_loss,
        val_loss=val_loss,
        train_acc=train_acc,
        val_acc=val_acc,
        tta_val_acc=tta_val_acc,
        name=name,
        batch_size=batch_size,
        muon_lr=muon_lr,
        use_best_lr=use_best_lr,
        best_lr_strategy=best_lr_strategy,
        best_lr_rel_diff_threshold=best_lr_rel_diff_threshold,
        best_lr_linear_decay=best_lr_linear_decay,
        best_lr_scheduler=best_lr_scheduler,
        best_lr_ema=best_lr_ema,
        best_lr_logs=best_lr_logs,
        sgd_lr_mult=SGD_LR_MULT,
        training_batch_loss_logs=training_batch_loss_logs,
        training_batch_loss_log_steps=loss_log_steps_list,
        wall_time_seconds=wall_time_seconds,
        cuda_time_seconds=time_seconds,
    )


if __name__ == "__main__":
    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    results = []
    run_index = [0]

    def evaluate_point(batch_size, point, result_cache):
        cache_key = schedule_cache_key(point)
        if cache_key in result_cache:
            cached = result_cache[cache_key]
            print(
                "schedule_search cache_hit batch_size=%d point=%s cache_key=%s "
                "step1_lr=%.6g step5_lr=%.6g step10_lr=%.6g train_loss=%.4f"
                % (
                    batch_size,
                    point,
                    cache_key,
                    cache_key[0],
                    cache_key[1],
                    cache_key[2],
                    cached["train_loss"],
                ),
                flush=True,
            )
            return cached

        config = overfit_run_config(batch_size, point)
        print(
            "cifar_baseline2 run=%d batch_size=%d muon_lr=%.6g "
            "sgd_lr_mult=%.6g name=%s best_lr_strategy=%s "
            "best_lr_linear_decay=%s best_lr_scheduler=%s"
            % (
                run_index[0],
                config["batch_size"],
                config["muon_lr"],
                config["sgd_lr_mult"],
                config["name"],
                config["best_lr_strategy"],
                config.get("best_lr_linear_decay", False),
                resolve_best_lr_scheduler(
                    config.get("best_lr_scheduler"),
                    config.get("best_lr_linear_decay", False),
                ),
            ),
            flush=True,
        )
        print(
            "applied_lr_schedule name=%s point=%s step1_lr=%.8g "
            "step5_lr=%.8g step10_lr=%.8g values=%s"
            % (
                config["best_lr_scheduler"],
                point,
                schedule_cache_key(point)[0],
                schedule_cache_key(point)[1],
                schedule_cache_key(point)[2],
                ",".join(
                    "%.8g" % value
                    for value in APPLIED_LR_SCHEDULES[config["best_lr_scheduler"]]
                ),
            ),
            flush=True,
        )
        result = main(run_index[0], model, **config)
        result["schedule_search_point"] = point
        (
            result["schedule_step1_lr"],
            result["schedule_step5_lr"],
            result["schedule_step10_lr"],
        ) = schedule_cache_key(point)
        result["applied_lr_schedule_values"] = APPLIED_LR_SCHEDULES[
            config["best_lr_scheduler"]
        ]
        result_cache[cache_key] = result
        results.append(result)
        run_index[0] += 1

        print("Name:               %s" % result["name"])
        print("Batch size:         %d" % result["batch_size"])
        print("Muon lr:            %.6g" % result["muon_lr"])
        print("SGD lr mult:        %.6g" % result["sgd_lr_mult"])
        print("Use best lr:        %s" % result["use_best_lr"])
        print("Applied LR schedule:%s" % result["best_lr_scheduler"])
        if result["use_best_lr"]:
            print("Best lr strategy:   %s" % result["best_lr_strategy"])
            print("Best lr decay:      %s" % result["best_lr_linear_decay"])
            print("Best lr scheduler:  %s" % result["best_lr_scheduler"])
            print("Final best lr ema:  %.6g" % result["best_lr_ema"])
        print("Train loss:         %.4f" % result["train_loss"])
        print("Val loss:           %.4f" % result["val_loss"])
        print("Train acc:          %.4f" % result["train_acc"])
        print("Val acc:            %.4f" % result["val_acc"])
        print("TTA val acc:        %.4f" % result["tta_val_acc"])
        return result

    search_summaries = []
    for batch_size in OVERFIT_BATCH_SIZES:
        result_cache = {}
        current_point = SCHEDULE_SEARCH_INITIAL_POINT
        current_result = evaluate_point(batch_size, current_point, result_cache)

        for move in range(SCHEDULE_SEARCH_MAX_MOVES):
            neighbor_points = schedule_search_neighbors(current_point)
            neighbor_results = [
                evaluate_point(batch_size, point, result_cache)
                for point in neighbor_points
            ]
            candidates = [current_result] + neighbor_results
            best_result = min(candidates, key=lambda result: result["train_loss"])
            best_point = best_result["schedule_search_point"]
            print(
                "schedule_search step batch_size=%d move=%d center=%s "
                "center_train_loss=%.4f best=%s best_train_loss=%.4f "
                "neighbors=%d"
                % (
                    batch_size,
                    move,
                    current_point,
                    current_result["train_loss"],
                    best_point,
                    best_result["train_loss"],
                    len(neighbor_points),
                ),
                flush=True,
            )
            if best_point == current_point:
                break
            current_point = best_point
            current_result = best_result
        else:
            print(
                "schedule_search max_moves_reached batch_size=%d point=%s "
                "train_loss=%.4f"
                % (batch_size, current_point, current_result["train_loss"]),
                flush=True,
            )

        search_summaries.append(
            dict(
                batch_size=batch_size,
                best_point=current_point,
                best_result=current_result,
                evaluated_points=sorted(result_cache),
            )
        )
        print(
            "schedule_search complete batch_size=%d best_point=%s "
            "step1_lr=%.6g step5_lr=%.6g step10_lr=%.6g train_loss=%.4f "
            "evaluated_points=%d"
            % (
                batch_size,
                current_point,
                current_result["schedule_step1_lr"],
                current_result["schedule_step5_lr"],
                current_result["schedule_step10_lr"],
                current_result["train_loss"],
                len(result_cache),
            ),
            flush=True,
        )

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(
        dict(
            code=code,
            results=results,
            search_summaries=search_summaries,
            applied_lr_schedules=APPLIED_LR_SCHEDULES,
        ),
        log_path,
    )
    print(os.path.abspath(log_path))
