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

with open(sys.argv[0]) as f:
    code = f.read()
import json
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


def _env_flag(name):
    value = os.getenv(name)
    if value is None:
        return None
    return value.lower() not in {"0", "false", "no"}


def _cuda_capability():
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability(0)


CUDA_CAPABILITY = _cuda_capability()
IS_AMPERE_OR_NEWER = CUDA_CAPABILITY[0] >= 8
USE_CUDNN_BENCHMARK = bool(_env_flag("CIFAR_SPEEDRUN_CUDNN_BENCHMARK"))
USE_TF32 = (_env_flag("CIFAR_SPEEDRUN_TF32") is not False) and IS_AMPERE_OR_NEWER

torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.backends.cuda.matmul.allow_tf32 = USE_TF32

USE_COMPILED_MUON = (
    bool(_env_flag("CIFAR_SPEEDRUN_COMPILE_MUON")) and IS_AMPERE_OR_NEWER
)
MUON_DTYPE = torch.bfloat16 if IS_AMPERE_OR_NEWER else torch.float16

#############################################
#               Muon optimizer              #
#############################################


def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
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
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (
            (self.images.half() / 255)
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
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

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


def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ""
    for col in columns_list:
        print_string += "|  %s  " % col
    print_string += "|"
    if is_head:
        print("-" * len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-" * len(print_string))


logging_columns_list = [
    "run          ",
    "head_momentum",
    "muon_momentum",
    "epoch",
    "train_acc",
    "val_acc",
    "tta_val_acc",
    "time_seconds",
]


def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = "{:0.4f}".format(var)
        else:
            assert var is None
            res = ""
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)


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


############################################
#              Line Search                 #
############################################

LINE_SEARCH_FACTOR = 0.8
LINE_SEARCH_EMA_BETA = 0.9
LINE_SEARCH_MAX_ITERS = 50
BOUNDARY_SEARCH_MAX_ITERS = 50
LEFT_LANDSCAPE_STEPS = 10
PARAM_GROUP_NAMES = ("head", "muon")
MOMENTUM_VALUES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9)
LINE_SEARCH_LOG_CONTEXT = {}


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
        None if p.grad is None else p.grad.detach().clone(memory_format=torch.preserve_format)
        for p in model.parameters()
    ]


def restore_grads(model, grads):
    for p, grad in zip(model.parameters(), grads):
        p.grad = None if grad is None else grad.clone(memory_format=torch.preserve_format)


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


def compute_train_loss(model, inputs, labels, whiten_bias_grad):
    outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
    loss = F.cross_entropy(outputs.float(), labels, label_smoothing=0.2, reduction="sum")
    return outputs, loss


def finite_loss_item(loss):
    return loss.item() if torch.isfinite(loss) else float("inf")


def lrs_to_dict(lrs):
    return {name: float(lr) for name, lr in zip(PARAM_GROUP_NAMES, lrs)}


def emit_line_search_log(**payload):
    payload = dict(LINE_SEARCH_LOG_CONTEXT, **payload)
    payload["event"] = "lr_landscape"
    print("LR_LANDSCAPE " + json.dumps(payload, sort_keys=True), flush=True)


def compute_loss_item(model, inputs, labels, whiten_bias_grad):
    with torch.no_grad():
        _, loss = compute_train_loss(model, inputs, labels, whiten_bias_grad)
    return finite_loss_item(loss)


def compute_loss_item_preserving_state(
    model, optimizers, inputs, labels, whiten_bias_grad
):
    state = capture_training_state(model, optimizers)
    loss = compute_loss_item(model, inputs, labels, whiten_bias_grad)
    restore_training_state(model, optimizers, state)
    return loss


def lr_tuple_from_exponents(base_lrs, exponents):
    return tuple(
        base_lr * (LINE_SEARCH_FACTOR**exponent)
        for base_lr, exponent in zip(base_lrs, exponents)
    )


def update_lr_ema(ema_lrs, line_search_lrs):
    if ema_lrs is None:
        return line_search_lrs
    return tuple(
        LINE_SEARCH_EMA_BETA * ema_lr + (1 - LINE_SEARCH_EMA_BETA) * line_search_lr
        for ema_lr, line_search_lr in zip(ema_lrs, line_search_lrs)
    )


def set_line_search_lrs(optimizer1, optimizer2, lrs):
    head_lr, optimizer2_lr = lrs
    optimizer1.param_groups[0]["lr"] = head_lr
    for group in optimizer2.param_groups:
        group["lr"] = optimizer2_lr


def line_search(
    model,
    optimizers,
    optimizer1,
    optimizer2,
    inputs,
    labels,
    eval_inputs,
    eval_labels,
    whiten_bias_grad,
    base_lrs,
    step,
):
    search_state = capture_training_state(model, optimizers)
    losses = {}
    records = {}

    def restore_search_state():
        restore_training_state(model, optimizers, search_state)

    def evaluate_candidate(exponents):
        if exponents in losses:
            return losses[exponents]
        lrs = lr_tuple_from_exponents(base_lrs, exponents)
        restore_search_state()
        set_line_search_lrs(optimizer1, optimizer2, lrs)
        for opt in optimizers:
            opt.step()
        post_step_state = capture_training_state(model, optimizers)
        train_loss = compute_loss_item(model, inputs, labels, whiten_bias_grad)
        restore_training_state(model, optimizers, post_step_state)
        eval_loss = compute_loss_item(model, eval_inputs, eval_labels, whiten_bias_grad)
        losses[exponents] = train_loss
        records[exponents] = dict(
            exponents=exponents,
            lrs=lrs,
            train_loss=train_loss,
            eval_loss=eval_loss,
        )
        emit_line_search_log(
            step=step,
            phase="coordinate_search",
            exponents=list(exponents),
            lrs=lrs_to_dict(lrs),
            train_loss=train_loss,
            eval_loss=eval_loss,
        )
        return train_loss

    center = (0,) * len(base_lrs)
    for _ in range(LINE_SEARCH_MAX_ITERS):
        candidates = [center]
        for dim in range(len(center)):
            for offset in (-1, 1):
                candidate = list(center)
                candidate[dim] += offset
                candidates.append(tuple(candidate))
        for candidate in candidates:
            evaluate_candidate(candidate)

        best_exponents = min(candidates, key=lambda candidate: losses[candidate])
        if best_exponents == center:
            restore_search_state()
            best_record = records[center]
            return (
                best_record["lrs"],
                best_record["train_loss"],
                best_record["eval_loss"],
            )
        center = best_exponents

    restore_search_state()
    best_exponents = min(losses, key=lambda candidate: losses[candidate])
    best_record = records[best_exponents]
    return best_record["lrs"], best_record["train_loss"], best_record["eval_loss"]


def evaluate_lrs_from_state(
    model,
    optimizers,
    optimizer1,
    optimizer2,
    search_state,
    inputs,
    labels,
    eval_inputs,
    eval_labels,
    whiten_bias_grad,
    lrs,
):
    restore_training_state(model, optimizers, search_state)
    set_line_search_lrs(optimizer1, optimizer2, lrs)
    for opt in optimizers:
        opt.step()
    post_step_state = capture_training_state(model, optimizers)
    train_loss = compute_loss_item(model, inputs, labels, whiten_bias_grad)
    restore_training_state(model, optimizers, post_step_state)
    eval_loss = compute_loss_item(model, eval_inputs, eval_labels, whiten_bias_grad)
    return train_loss, eval_loss


def log_line_search_landscape(
    model,
    optimizers,
    optimizer1,
    optimizer2,
    inputs,
    labels,
    eval_inputs,
    eval_labels,
    whiten_bias_grad,
    best_lrs,
    pre_train_loss,
    pre_eval_loss,
    best_train_loss,
    best_eval_loss,
    step,
):
    search_state = capture_training_state(model, optimizers)

    for dim, group_name in enumerate(PARAM_GROUP_NAMES):
        left_lrs = list(best_lrs)
        left_lrs[dim] = 0.0
        train_loss, eval_loss = evaluate_lrs_from_state(
            model,
            optimizers,
            optimizer1,
            optimizer2,
            search_state,
            inputs,
            labels,
            eval_inputs,
            eval_labels,
            whiten_bias_grad,
            tuple(left_lrs),
        )
        emit_line_search_log(
            step=step,
            phase="landscape",
            group=group_name,
            point="left_origin",
            varied_lr=0.0,
            lrs=lrs_to_dict(left_lrs),
            train_loss=train_loss,
            eval_loss=eval_loss,
            pre_train_loss=pre_train_loss,
            pre_eval_loss=pre_eval_loss,
        )
        for left_iter in range(LEFT_LANDSCAPE_STEPS, 0, -1):
            left_lrs = list(best_lrs)
            left_lrs[dim] = best_lrs[dim] * (LINE_SEARCH_FACTOR**left_iter)
            train_loss, eval_loss = evaluate_lrs_from_state(
                model,
                optimizers,
                optimizer1,
                optimizer2,
                search_state,
                inputs,
                labels,
                eval_inputs,
                eval_labels,
                whiten_bias_grad,
                tuple(left_lrs),
            )
            emit_line_search_log(
                step=step,
                phase="landscape",
                group=group_name,
                point="left_probe",
                left_iter=left_iter,
                varied_lr=float(left_lrs[dim]),
                lrs=lrs_to_dict(left_lrs),
                train_loss=train_loss,
                eval_loss=eval_loss,
                pre_train_loss=pre_train_loss,
                pre_eval_loss=pre_eval_loss,
            )
        emit_line_search_log(
            step=step,
            phase="landscape",
            group=group_name,
            point="optimum",
            varied_lr=float(best_lrs[dim]),
            lrs=lrs_to_dict(best_lrs),
            train_loss=best_train_loss,
            eval_loss=best_eval_loss,
            pre_train_loss=pre_train_loss,
            pre_eval_loss=pre_eval_loss,
        )

        probe_lrs = list(best_lrs)
        boundary_found = best_train_loss >= pre_train_loss
        for boundary_iter in range(BOUNDARY_SEARCH_MAX_ITERS):
            if boundary_found:
                break
            probe_lrs[dim] = probe_lrs[dim] / LINE_SEARCH_FACTOR
            train_loss, eval_loss = evaluate_lrs_from_state(
                model,
                optimizers,
                optimizer1,
                optimizer2,
                search_state,
                inputs,
                labels,
                eval_inputs,
                eval_labels,
                whiten_bias_grad,
                tuple(probe_lrs),
            )
            boundary_found = train_loss >= pre_train_loss
            emit_line_search_log(
                step=step,
                phase="landscape",
                group=group_name,
                point="boundary" if boundary_found else "probe",
                boundary_found=boundary_found,
                boundary_iter=boundary_iter + 1,
                varied_lr=float(probe_lrs[dim]),
                lrs=lrs_to_dict(probe_lrs),
                train_loss=train_loss,
                eval_loss=eval_loss,
                pre_train_loss=pre_train_loss,
                pre_eval_loss=pre_eval_loss,
            )

        if not boundary_found:
            emit_line_search_log(
                step=step,
                phase="landscape",
                group=group_name,
                point="boundary_not_found",
                boundary_found=False,
                varied_lr=float(probe_lrs[dim]),
                lrs=lrs_to_dict(probe_lrs),
                train_loss=train_loss,
                eval_loss=eval_loss,
                pre_train_loss=pre_train_loss,
                pre_eval_loss=pre_eval_loss,
            )

    restore_training_state(model, optimizers, search_state)


############################################
#                Training                  #
############################################


def main(run, model, head_momentum, muon_momentum):
    global LINE_SEARCH_LOG_CONTEXT

    batch_size = 2000
    head_lr = 0.67
    wd = 2e-6 * batch_size
    total_train_steps = 20
    LINE_SEARCH_LOG_CONTEXT = dict(
        run=run,
        head_momentum=float(head_momentum),
        muon_momentum=float(muon_momentum),
    )

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    train_loader.shuffle = False
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
    fixed_train_iter = iter(train_loader)
    train_inputs, train_labels = next(fixed_train_iter)
    eval_inputs, eval_labels = next(fixed_train_iter)

    # Create optimizers and learning rate schedulers
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    model.whiten.bias.requires_grad = False
    for p in norm_biases:
        p.requires_grad = False
    param_configs = [
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs,
        momentum=head_momentum,
        nesterov=head_momentum > 0,
        fused=True,
    )
    optimizer2 = Muon(
        filter_params, lr=0.24, momentum=muon_momentum, nesterov=muon_momentum > 0
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
    ema_line_search_lrs = None

    # The first fixed augmented batch is the whole training dataset for this run.
    start_timer()
    model.init_whiten(train_inputs)
    stop_timer()

    for step in range(total_train_steps):
        start_timer()
        model.train()
        whiten_bias_grad = False
        outputs, train_loss = compute_train_loss(
            model, train_inputs, train_labels, whiten_bias_grad
        )
        train_loss.backward()
        pre_train_loss = finite_loss_item(train_loss)
        pre_eval_loss = compute_loss_item_preserving_state(
            model, optimizers, eval_inputs, eval_labels, whiten_bias_grad
        )
        line_search_base_lrs = (
            tuple(group["initial_lr"] for group in optimizer1.param_groups)
            + (optimizer2.param_groups[0]["initial_lr"],)
            if ema_line_search_lrs is None
            else ema_line_search_lrs
        )
        emit_line_search_log(
            step=step,
            phase="pre_step",
            lrs=lrs_to_dict((0.0,) * len(PARAM_GROUP_NAMES)),
            train_loss=pre_train_loss,
            eval_loss=pre_eval_loss,
        )
        line_search_lrs, line_search_loss, line_search_eval_loss = line_search(
            model,
            optimizers,
            optimizer1,
            optimizer2,
            train_inputs,
            train_labels,
            eval_inputs,
            eval_labels,
            whiten_bias_grad,
            line_search_base_lrs,
            step,
        )
        log_line_search_landscape(
            model,
            optimizers,
            optimizer1,
            optimizer2,
            train_inputs,
            train_labels,
            eval_inputs,
            eval_labels,
            whiten_bias_grad,
            line_search_lrs,
            pre_train_loss,
            pre_eval_loss,
            line_search_loss,
            line_search_eval_loss,
            step,
        )
        ema_line_search_lrs = update_lr_ema(ema_line_search_lrs, line_search_lrs)
        line_search_head_lr, line_search_optimizer2_lr = line_search_lrs
        print(
            "line_search step=%d "
            "line_search_head_lr=%.8g line_search_optimizer2_lr=%.8g "
            "pre_train_loss=%.6f pre_eval_loss=%.6f "
            "line_search_loss=%.6f line_search_eval_loss=%.6f"
            % (
                step,
                line_search_head_lr,
                line_search_optimizer2_lr,
                pre_train_loss,
                pre_eval_loss,
                line_search_loss,
                line_search_eval_loss,
            )
        )
        set_line_search_lrs(optimizer1, optimizer2, line_search_lrs)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        stop_timer()

    train_acc = (outputs.detach().argmax(1) == train_labels).float().mean().item()
    val_acc = evaluate(model, test_loader, tta_level=0)
    epoch = "20step"
    print_training_details(locals(), is_final_entry=False)
    run = None  # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc


if __name__ == "__main__":
    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    print_columns(logging_columns_list, is_head=True)
    # main("warmup", model)
    momentum_configs = [
        (head_momentum, muon_momentum)
        for head_momentum in MOMENTUM_VALUES
        for muon_momentum in MOMENTUM_VALUES
    ]
    accs = torch.tensor(
        [
            main(
                f"h{head_momentum:g}_m{muon_momentum:g}",
                model,
                head_momentum,
                muon_momentum,
            )
            for head_momentum, muon_momentum in momentum_configs
        ]
    )
    print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(
        dict(code=code, accs=accs, momentum_configs=momentum_configs), log_path
    )
    print(os.path.abspath(log_path))
