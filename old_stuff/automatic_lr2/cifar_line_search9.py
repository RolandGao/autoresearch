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
from collections import Counter
import json
import uuid
from math import ceil, isfinite

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
TRAINING_SEED = 0


def set_training_seed():
    torch.manual_seed(TRAINING_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAINING_SEED)


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


logging_columns_list = [
    "run",
    "epoch",
    "train_loss",
    "train_acc",
    "val_acc",
    "tta_val_acc",
    "time_seconds",
]


def print_training_details(variables, is_final_entry):
    fields = []
    for col in logging_columns_list:
        var = variables.get(col, None)
        if type(var) in (int, str):
            fields.append(f"{col}={var}")
        elif type(var) is float:
            fields.append(f"{col}={var:0.4f}")
        else:
            assert var is None
    print("epoch_summary " + " ".join(fields), flush=True)


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
INITIAL_GROUND_TRUTH_MATRIX_LR = 0.24
LINE_SEARCH_LEFT_STEPS = 30
LINE_SEARCH_RIGHT_STEPS = 20


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


def lr_from_exponent(base_lr, exponent):
    return base_lr * (LINE_SEARCH_FACTOR**exponent)


def set_matrix_lr(optimizer, matrix_lr):
    for group in optimizer.param_groups:
        group["lr"] = matrix_lr


def compute_line_search_losses(model, batches):
    losses = []
    model.train()
    for inputs, labels in batches:
        outputs = model(inputs)
        batch_losses = F.cross_entropy(
            outputs.float(), labels, label_smoothing=0.2, reduction="none"
        )
        losses.append(batch_losses)
    return torch.cat(losses)


def finite_loss_item(loss):
    return loss.item() if torch.isfinite(loss) else float("inf")


def finite_loss_sum_item(losses):
    return finite_loss_item(losses.sum())


def finite_float_or_none(value):
    value = float(value)
    return value if isfinite(value) else None


def sample_lr_loss_curve_payload(sample_losses_by_matrix_lr):
    matrix_lrs = sorted(sample_losses_by_matrix_lr)
    if not matrix_lrs:
        return dict(lrs=[], optimal_lr_counts={}, curves=[])
    losses = torch.stack(
        [
            torch.where(
                torch.isfinite(sample_losses_by_matrix_lr[matrix_lr]),
                sample_losses_by_matrix_lr[matrix_lr],
                torch.full_like(sample_losses_by_matrix_lr[matrix_lr], float("inf")),
            )
            for matrix_lr in matrix_lrs
        ]
    )
    best_indices = losses.argmin(dim=0).detach().cpu().tolist()
    best_lrs = [matrix_lrs[index] for index in best_indices]
    optimal_lr_counts = Counter(best_lrs)

    selected_indices = []
    seen_best_indices = set()
    for sample_index, best_index in enumerate(best_indices):
        if best_index in seen_best_indices:
            continue
        selected_indices.append(sample_index)
        seen_best_indices.add(best_index)

    losses = losses.detach().cpu()
    return dict(
        lrs=format_matrix_lrs(matrix_lrs),
        optimal_lr_counts={
            "%.8g" % matrix_lr: count
            for matrix_lr, count in sorted(optimal_lr_counts.items())
        },
        curves=[
            dict(
                sample_index=sample_index,
                optimal_lr=float("%.8g" % best_lrs[sample_index]),
                losses=[
                    finite_float_or_none(losses[lr_index, sample_index].item())
                    for lr_index in range(len(matrix_lrs))
                ],
            )
            for sample_index in selected_indices
        ],
    )


def evaluate_matrix_lr_grid(
    model,
    optimizers,
    matrix_optimizer,
    validation_sets,
    matrix_lrs,
):
    search_state = capture_training_state(model, optimizers)
    losses_by_name = {name: {} for name, _ in validation_sets}
    sample_losses_by_name = {name: {} for name, _ in validation_sets}

    def restore_search_state():
        restore_training_state(model, optimizers, search_state)

    for matrix_lr in matrix_lrs:
        if all(matrix_lr in losses for losses in losses_by_name.values()):
            continue
        for name, batches in validation_sets:
            if matrix_lr in losses_by_name[name]:
                continue
            restore_search_state()
            set_matrix_lr(matrix_optimizer, matrix_lr)
            for opt in optimizers:
                opt.step()
            with torch.no_grad():
                losses = compute_line_search_losses(model, batches)
            losses_by_name[name][matrix_lr] = finite_loss_sum_item(losses)
            if name == "current_batch":
                sample_losses_by_name[name][matrix_lr] = losses.detach()

    restore_search_state()
    results = {}
    for name, losses in losses_by_name.items():
        results[name] = dict(
            best_matrix_lr=min(losses.items(), key=lambda item: item[1])[0],
            best_loss=min(losses.values()),
            losses_by_matrix_lr=losses,
        )
        sample_losses = sample_losses_by_name.get(name, {})
        if sample_losses:
            results[name]["sample_lr_loss_curves"] = sample_lr_loss_curve_payload(
                sample_losses
            )
    return results


def initial_ground_truth_matrix_lr_sweep(ground_truth_matrix_lr):
    sweep = [0.0, INITIAL_GROUND_TRUTH_MATRIX_LR]
    for i in range(1, LINE_SEARCH_LEFT_STEPS + 1):
        sweep.append(lr_from_exponent(INITIAL_GROUND_TRUTH_MATRIX_LR, i))
    for i in range(1, LINE_SEARCH_RIGHT_STEPS + 1):
        sweep.append(lr_from_exponent(INITIAL_GROUND_TRUTH_MATRIX_LR, -i))
    sweep.append(ground_truth_matrix_lr)
    return sweep


def format_matrix_lr_loss_map(losses_by_matrix_lr):
    return {
        "%.8g" % matrix_lr: loss
        for matrix_lr, loss in sorted(losses_by_matrix_lr.items())
    }


def format_matrix_lrs(matrix_lrs):
    return [float("%.8g" % matrix_lr) for matrix_lr in matrix_lrs]


def line_search_validation_sets(epoch_batches, batch_index):
    return [
        ("current_batch", [epoch_batches[batch_index]]),
    ]


############################################
#                Training                  #
############################################


def main(run, model):
    set_training_seed()

    batch_size = 125
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2)
    )
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    param_configs = [
        dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd / head_lr),
    ]
    optimizer1 = torch.optim.SGD(
        param_configs, momentum=0.85, nesterov=True, fused=True
    )
    optimizer2 = Muon(filter_params, lr=0.04, momentum=0.6, nesterov=True)
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

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        epoch_batches = list(train_loader)
        for batch_index, (inputs, labels) in enumerate(epoch_batches):
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            train_loss = F.cross_entropy(
                outputs.float(), labels, label_smoothing=0.2, reduction="sum"
            )
            train_loss.backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:] + optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            ground_truth_matrix_lr = optimizer2.param_groups[0]["lr"]
            line_search_results = evaluate_matrix_lr_grid(
                model,
                optimizers,
                optimizer2,
                line_search_validation_sets(epoch_batches, batch_index),
                initial_ground_truth_matrix_lr_sweep(ground_truth_matrix_lr),
            )
            current_batch_results = line_search_results["current_batch"]
            print(
                "\nline_search step=%d "
                "line_search_matrix_lr=%.8g ground_truth_matrix_lr=%.8g "
                "pre_update_train_loss=%.6f line_search_loss=%.6f "
                "current_batch_line_search_matrix_lr=%.8g "
                "current_batch_line_search_loss=%.6f "
                "current_batch_matrix_lr_train_losses=%s "
                "current_batch_sample_lr_loss_curves=%s"
                % (
                    step,
                    current_batch_results["best_matrix_lr"],
                    ground_truth_matrix_lr,
                    finite_loss_item(train_loss),
                    current_batch_results["best_loss"],
                    current_batch_results["best_matrix_lr"],
                    current_batch_results["best_loss"],
                    json.dumps(
                        format_matrix_lr_loss_map(
                            current_batch_results["losses_by_matrix_lr"]
                        ),
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    json.dumps(
                        current_batch_results["sample_lr_loss_curves"],
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                ),
                flush=True,
            )
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_loss = F.cross_entropy(
            outputs.detach().float(), labels, label_smoothing=0.2
        ).item()
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)

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
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    # main("warmup", model)
    accs = torch.tensor([main(run, model) for run in range(1)])
    print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, accs=accs), log_path)
    print(os.path.abspath(log_path))
