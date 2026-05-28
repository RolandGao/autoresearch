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


def log_step_train_loss(step, total_steps, name, train_loss):
    print(
        f"step_train_loss step={step}/{total_steps} name={name} "
        f"train_loss={repr(float(train_loss))}",
        flush=True,
    )


def log_search_train_loss(
    run,
    interval,
    train_step,
    candidate_step,
    candidate_steps,
    k,
    lr,
    train_loss,
):
    print(
        "n_search_train_loss "
        f"run={run} interval={interval} train_step={train_step} "
        f"candidate_step={candidate_step}/{candidate_steps} "
        f"k={k} lr={lr:.8g} train_loss={repr(float(train_loss))}",
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
OVERFIT_INTERVAL_STEPS_LIST = [1, 2, 5, 10, 20]
OVERFIT_TRAIN_STEPS = 100
OVERFIT_INITIAL_LR_K = 0
OVERFIT_BASE_LR = 0.2
OVERFIT_LR_FACTOR = 0.6
OVERFIT_MAX_LR_SEARCH_MOVES = 40
OVERFIT_MUON_MOMENTUM = 0.6
OVERFIT_SGD_LR_MULT = 1.0


def round_sigfigs(value, sigfigs=2):
    if value == 0:
        return 0.0
    return round(value, sigfigs - 1 - floor(log10(abs(value))))


def lr_from_k(k):
    return round_sigfigs(OVERFIT_BASE_LR * (OVERFIT_LR_FACTOR**k), 2)


def lr_label(lr):
    return ("%.8g" % lr).replace("-", "m").replace(".", "p")


def overfit_run_config(batch_size, interval_steps):
    return dict(
        name=f"n_search_bs{batch_size}_N{interval_steps}",
        batch_size=batch_size,
        train_steps=OVERFIT_TRAIN_STEPS,
        interval_steps=interval_steps,
        initial_lr_k=OVERFIT_INITIAL_LR_K,
        muon_momentum=OVERFIT_MUON_MOMENTUM,
        sgd_lr_mult=OVERFIT_SGD_LR_MULT,
    )


def train_one_step(
    model,
    optimizers,
    muon_optimizer,
    inputs,
    labels,
    whiten_bias_grad,
    muon_lr,
):
    set_muon_lr(muon_optimizer, muon_lr)
    model.train()
    outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
    loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="mean")
    loss.backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)


def main(
    run,
    model,
    name,
    batch_size,
    train_steps,
    interval_steps,
    initial_lr_k,
    muon_momentum,
    sgd_lr_mult=None,
):
    run_id = run
    run_wall_start = time.perf_counter()
    set_training_seed()

    SGD_LR_MULT = OVERFIT_SGD_LR_MULT if sgd_lr_mult is None else sgd_lr_mult
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
    total_train_steps = train_steps
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
    optimizer2 = Muon(
        filter_params,
        lr=lr_from_k(initial_lr_k),
        momentum=muon_momentum,
        nesterov=muon_momentum > 0,
    )
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
    step_train_loss_logs = []
    interval_search_logs = []
    selected_lrs = []
    selected_lr_ks = []
    inputs, labels = training_batches[0]
    current_k = initial_lr_k
    interval_index = 0

    while step < total_train_steps:
        interval_index += 1
        interval_start_step = step
        steps_this_interval = min(interval_steps, total_train_steps - step)
        interval_start_state = capture_training_state(model, optimizers)
        candidate_cache = {}

        def evaluate_candidate(k):
            lr = lr_from_k(k)
            if lr in candidate_cache:
                cached = candidate_cache[lr]
                print(
                    "n_search cache_hit run=%s interval=%d start_step=%d "
                    "N=%d k=%d lr=%.8g train_loss=%s"
                    % (
                        run_id,
                        interval_index,
                        interval_start_step,
                        steps_this_interval,
                        k,
                        lr,
                        repr(float(cached["train_loss"])),
                    ),
                    flush=True,
                )
                cached_for_k = dict(cached)
                cached_for_k["k"] = k
                return cached_for_k

            restore_training_state(model, optimizers, interval_start_state)
            local_losses = []
            start_timer()
            for offset in range(steps_this_interval):
                global_step = interval_start_step + offset
                train_one_step(
                    model,
                    optimizers,
                    optimizer2,
                    inputs,
                    labels,
                    global_step < whiten_bias_train_steps,
                    lr,
                )
                train_loss = evaluate_training_batch_losses(model, [(inputs, labels)])[0]
                local_losses.append(train_loss)
                log_search_train_loss(
                    run_id,
                    interval_index,
                    global_step + 1,
                    offset + 1,
                    steps_this_interval,
                    k,
                    lr,
                    train_loss,
                )
            stop_timer()
            train_loss = local_losses[-1]
            candidate_state = capture_training_state(model, optimizers)
            candidate = dict(
                k=k,
                lr=lr,
                train_loss=train_loss,
                step_train_losses=local_losses,
                state=candidate_state,
            )
            candidate_cache[lr] = candidate
            print(
                "n_search candidate run=%s interval=%d start_step=%d "
                "steps=%d k=%d lr=%.8g train_loss=%s"
                % (
                    run_id,
                    interval_index,
                    interval_start_step,
                    steps_this_interval,
                    k,
                    lr,
                    repr(float(train_loss)),
                ),
                flush=True,
            )
            return candidate

        search_moves = 0
        while True:
            search_moves += 1
            if search_moves > OVERFIT_MAX_LR_SEARCH_MOVES:
                raise RuntimeError(
                    "LR interval search did not converge: "
                    f"run={run_id} interval={interval_index} center_k={current_k}"
                )
            center = evaluate_candidate(current_k)
            lower_lr = evaluate_candidate(current_k + 1)
            higher_lr = evaluate_candidate(current_k - 1)
            candidates = [center, lower_lr, higher_lr]
            best = min(candidates, key=lambda candidate: candidate["train_loss"])
            print(
                "n_search step run=%s interval=%d center_k=%d center_lr=%.8g "
                "center_loss=%s best_k=%d best_lr=%.8g best_loss=%s"
                % (
                    run_id,
                    interval_index,
                    current_k,
                    center["lr"],
                    repr(float(center["train_loss"])),
                    best["k"],
                    best["lr"],
                    repr(float(best["train_loss"])),
                ),
                flush=True,
            )
            if best["k"] == current_k:
                selected = center
                break
            current_k = best["k"]

        restore_training_state(model, optimizers, selected["state"])
        for offset, train_loss_value in enumerate(selected["step_train_losses"], start=1):
            committed_step = interval_start_step + offset
            log_applied_lr(committed_step, total_train_steps, name, selected["lr"])
            step_train_loss_logs.append(
                dict(
                    step=committed_step,
                    train_loss=train_loss_value,
                    interval=interval_index,
                    lr=selected["lr"],
                    lr_k=selected["k"],
                )
            )
            log_step_train_loss(
                committed_step, total_train_steps, name, train_loss_value
            )

        step += steps_this_interval
        selected_lrs.append(selected["lr"])
        selected_lr_ks.append(selected["k"])
        interval_log = dict(
            interval=interval_index,
            start_step=interval_start_step + 1,
            end_step=step,
            interval_steps=steps_this_interval,
            selected_k=selected["k"],
            selected_lr=selected["lr"],
            train_loss=selected["train_loss"],
            evaluated_candidates=sorted(
                (
                    dict(
                        k=value["k"],
                        lr=lr,
                        train_loss=value["train_loss"],
                        step_train_losses=value["step_train_losses"],
                    )
                    for lr, value in candidate_cache.items()
                ),
                key=lambda row: row["k"],
            ),
        )
        interval_search_logs.append(interval_log)
        print(
            "n_search interval_selected run=%s interval=%d steps=%d-%d "
            "N=%d selected_k=%d selected_lr=%.8g train_loss=%s "
            "evaluated_candidates=%d"
            % (
                run_id,
                interval_index,
                interval_log["start_step"],
                interval_log["end_step"],
                steps_this_interval,
                selected["k"],
                selected["lr"],
                repr(float(selected["train_loss"])),
                len(candidate_cache),
            ),
            flush=True,
        )

        if step % len(train_loader) == 0 or step >= total_train_steps:
            val_acc = evaluate(model, test_loader, tta_level=0)
            log_eval(run, (step - 1) // len(train_loader), val_acc, time_seconds)
            run = None

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
        train_steps=train_steps,
        interval_steps=interval_steps,
        initial_lr_k=initial_lr_k,
        initial_lr=lr_from_k(initial_lr_k),
        selected_lr_ks=selected_lr_ks,
        selected_lrs=selected_lrs,
        muon_momentum=muon_momentum,
        muon_lr_schedule="n_search",
        sgd_lr_mult=SGD_LR_MULT,
        sgd_lr_schedule="constant",
        interval_search_logs=interval_search_logs,
        training_batch_loss_logs=training_batch_loss_logs,
        training_batch_loss_log_steps=loss_log_steps_list,
        step_train_loss_logs=step_train_loss_logs,
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

    def evaluate_config(batch_size, interval_steps):
        config = overfit_run_config(batch_size, interval_steps)
        print(
            "cifar_baseline2 run=%d train_steps=%d batch_size=%d "
            "N=%d initial_lr=%.8g initial_lr_k=%d muon_momentum=%.6g "
            "sgd_lr_mult=%.6g name=%s search=True "
            "muon_lr_schedule=n_search sgd_lr_schedule=constant"
            % (
                run_index[0],
                config["train_steps"],
                config["batch_size"],
                config["interval_steps"],
                lr_from_k(config["initial_lr_k"]),
                config["initial_lr_k"],
                config["muon_momentum"],
                config["sgd_lr_mult"],
                config["name"],
            ),
            flush=True,
        )
        result = main(run_index[0], model, **config)
        results.append(result)
        run_index[0] += 1

        print("Name:               %s" % result["name"])
        print("Train steps:        %d" % result["train_steps"])
        print("Batch size:         %d" % result["batch_size"])
        print("Interval steps:     %d" % result["interval_steps"])
        print("Initial Muon lr:    %.6g" % result["initial_lr"])
        print("Muon momentum:      %.6g" % result["muon_momentum"])
        print("SGD lr mult:        %.6g" % result["sgd_lr_mult"])
        print("Search:             True")
        print("Muon LR schedule:   n_search")
        print("SGD LR schedule:    constant")
        print(
            "Selected Muon lrs:  %s"
            % ",".join("%.8g" % value for value in result["selected_lrs"])
        )
        print("Train loss:         %.4f" % result["train_loss"])
        print("Val loss:           %.4f" % result["val_loss"])
        print("Train acc:          %.4f" % result["train_acc"])
        print("Val acc:            %.4f" % result["val_acc"])
        print("TTA val acc:        %.4f" % result["tta_val_acc"])
        return result

    run_summaries = []
    for batch_size in OVERFIT_BATCH_SIZES:
        for interval_steps in OVERFIT_INTERVAL_STEPS_LIST:
            result = evaluate_config(batch_size, interval_steps)
            run_summaries.append(
                dict(
                    batch_size=batch_size,
                    interval_steps=interval_steps,
                    selected_lr_ks=result["selected_lr_ks"],
                    selected_lrs=result["selected_lrs"],
                    muon_momentum=OVERFIT_MUON_MOMENTUM,
                    result=result,
                )
            )
            print(
                "n_search_run complete train_steps=%d batch_size=%d N=%d "
                "muon_momentum=%.6g train_loss=%.4f "
                "val_acc=%.4f tta_val_acc=%.4f"
                % (
                    OVERFIT_TRAIN_STEPS,
                    batch_size,
                    interval_steps,
                    OVERFIT_MUON_MOMENTUM,
                    result["train_loss"],
                    result["val_acc"],
                    result["tta_val_acc"],
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
            run_summaries=run_summaries,
            batch_sizes=OVERFIT_BATCH_SIZES,
            interval_steps_list=OVERFIT_INTERVAL_STEPS_LIST,
            initial_lr_k=OVERFIT_INITIAL_LR_K,
            initial_lr=lr_from_k(OVERFIT_INITIAL_LR_K),
            base_lr=OVERFIT_BASE_LR,
            lr_factor=OVERFIT_LR_FACTOR,
            muon_momentum=OVERFIT_MUON_MOMENTUM,
            train_steps=OVERFIT_TRAIN_STEPS,
        ),
        log_path,
    )
    print(os.path.abspath(log_path))
