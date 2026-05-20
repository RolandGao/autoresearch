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
import random
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

with open(sys.argv[0]) as f:
    code = f.read()
from itertools import product
import uuid
from math import ceil, exp

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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.backends.cuda.matmul.allow_tf32 = USE_TF32
torch.use_deterministic_algorithms(True)

USE_COMPILED_MUON = False
MUON_DTYPE = torch.bfloat16 if IS_AMPERE_OR_NEWER else torch.float16
TRAINING_SEED = 0


def set_training_seed():
    random.seed(TRAINING_SEED)
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
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(MUON_DTYPE)
    X /= (X.norm() + eps) # ensure top singular value <= 1
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

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    y = torch.arange(crop_size, device=images.device).view(1, crop_size, 1)
    x = torch.arange(crop_size, device=images.device).view(1, 1, crop_size)
    y_idx = y + (r + shifts[:, 0]).view(-1, 1, 1)
    x_idx = x + (r + shifts[:, 1]).view(-1, 1, 1)
    gathered_rows = images.gather(2, y_idx[:, None, :, :].expand(-1, images.size(1), -1, images.size(-1)))
    return gathered_rows.gather(3, x_idx[:, None, :, :].expand(-1, images.size(1), crop_size, crop_size))

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None):
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({"images": images, "labels": labels, "classes": dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device("cuda"))
        self.images, self.labels, self.classes = data["images"], data["labels"], data["classes"]
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
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
            self.proc_images["pad"] = F.pad(images, (pad,)*4, "reflect")

    def normalized_images(self):
        self._ensure_proc_images()
        return self.proc_images["norm"]

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

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

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

#############################################
#            Network Definition             #
#############################################

# note the use of low BatchNorm stats momentum
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding="same", bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
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
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths["block1"]),
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
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO="U")
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

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
        print("-"*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print("-"*len(print_string))

logging_columns_list = ["run   ", "epoch", "train_loss", "train_acc", "val_acc", "tta_val_acc", "time_seconds"]
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
        padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalized_images()
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.inference_mode():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

############################################
#                Training                  #
############################################

PIECEWISE_SCHEDULE_COUNT = 200
PIECEWISE_BOUNDARIES = (0, 5, 25, 175, 200)
PIECEWISE_SHAPES = ("linear", "power", "exp")
SEGMENT_POWER_CHOICES = (0.65, 0.80, 1.00, 1.25, 1.50, 1.85)
MIN_EXP_LR = 1e-5


def one_piece_lr(shape, initial_lr, parameter, step, total_train_steps=200):
    x = min(max(step / total_train_steps, 0.0), 1.0)
    if shape == "linear":
        multiplier = 1 - x
    elif shape == "power":
        multiplier = (1 - x) ** parameter
    elif shape == "exp":
        multiplier = exp(-parameter * x)
    else:
        raise ValueError("Unknown shape: %s" % shape)
    return initial_lr * multiplier


def boundary_profile(name, shape, initial_lr, parameter=None):
    return dict(
        name=name,
        boundaries=[
            one_piece_lr(shape, initial_lr, parameter, step)
            for step in PIECEWISE_BOUNDARIES
        ],
        preferred_shape=shape,
    )


def base_piecewise_profiles():
    return [
        boundary_profile("linear_0.31", "linear", 0.31),
        boundary_profile("linear_0.30", "linear", 0.30),
        boundary_profile("linear_0.27", "linear", 0.27),
        boundary_profile("linear_0.24", "linear", 0.24),
        boundary_profile("power_0.38_p1.25", "power", 0.38, 1.25),
        boundary_profile("power_0.16_p0.80", "power", 0.16, 0.80),
        boundary_profile("power_0.28_p1.25", "power", 0.28, 1.25),
        boundary_profile("power_0.42_p1.25", "power", 0.42, 1.25),
        boundary_profile("exp_0.45_a3.60", "exp", 0.45, 3.60),
        boundary_profile("exp_0.40_a3.30", "exp", 0.40, 3.30),
    ]


def clamp_lr(lr):
    return min(max(lr, 0.0), 0.70)


def jitter_lr(rng, base_lr, scale):
    if base_lr <= 0.002:
        return rng.choice((0.0, 0.001, 0.0025, 0.005, 0.008, 0.012))
    return clamp_lr(base_lr * rng.uniform(1 - scale, 1 + scale) + rng.uniform(-0.006, 0.006))


def make_segment(shape, start_lr, end_lr, rng, start_step, end_step):
    segment = dict(
        shape=shape,
        start_lr=round(start_lr, 8),
        end_lr=round(end_lr, 8),
        start_step=start_step,
        end_step=end_step,
    )
    if shape == "power":
        segment["power"] = rng.choice(SEGMENT_POWER_CHOICES)
    return segment


def continuous_piecewise_schedule(profile):
    segments = []
    for segment_index, shape in enumerate([profile["preferred_shape"]] * 4):
        start_step = PIECEWISE_BOUNDARIES[segment_index]
        end_step = PIECEWISE_BOUNDARIES[segment_index + 1]
        start_lr = profile["boundaries"][segment_index]
        end_lr = profile["boundaries"][segment_index + 1]
        segment = dict(
            shape=shape,
            start_lr=round(start_lr, 8),
            end_lr=round(end_lr, 8),
            start_step=start_step,
            end_step=end_step,
        )
        if shape == "power":
            segment["power"] = 1.25 if "p1.25" in profile["name"] else 0.80
        segments.append(segment)
    return dict(profile=profile["name"], segments=segments)


def make_random_piecewise_schedule(rng, profile, shape_pattern, jitter_scale):
    segments = []
    for segment_index, shape in enumerate(shape_pattern):
        start_step = PIECEWISE_BOUNDARIES[segment_index]
        end_step = PIECEWISE_BOUNDARIES[segment_index + 1]
        start_lr = jitter_lr(rng, profile["boundaries"][segment_index], jitter_scale)
        end_lr = jitter_lr(rng, profile["boundaries"][segment_index + 1], jitter_scale)
        segments.append(make_segment(shape, start_lr, end_lr, rng, start_step, end_step))
    return dict(profile=profile["name"], segments=segments)


def shape_pattern(schedule):
    return "".join(segment["shape"][0].upper() for segment in schedule["segments"])


def format_segment(segment):
    if segment["shape"] == "power":
        return "%s %.4g->%.4g p=%.3g" % (
            segment["shape"][0],
            segment["start_lr"],
            segment["end_lr"],
            segment["power"],
        )
    return "%s %.4g->%.4g" % (
        segment["shape"][0],
        segment["start_lr"],
        segment["end_lr"],
    )


def format_muon_schedule(schedule):
    return "%s %s" % (
        shape_pattern(schedule),
        " | ".join(format_segment(segment) for segment in schedule["segments"]),
    )


def make_muon_schedules():
    rng = random.Random(20260519)
    profiles = base_piecewise_profiles()
    schedules = [continuous_piecewise_schedule(profile) for profile in profiles]
    patterns = list(product(PIECEWISE_SHAPES, repeat=4))
    pattern_offset = 17
    while len(schedules) < PIECEWISE_SCHEDULE_COUNT:
        i = len(schedules) - len(profiles)
        profile = profiles[(i * 7) % len(profiles)]
        pattern = patterns[(i * 37 + pattern_offset) % len(patterns)]
        jitter_scale = 0.05 + 0.02 * ((i // len(patterns)) % 4)
        schedules.append(make_random_piecewise_schedule(rng, profile, pattern, jitter_scale))

    assert len(schedules) == PIECEWISE_SCHEDULE_COUNT
    for index, schedule in enumerate(schedules, start=1):
        schedule["index"] = index
        schedule["initial_lr"] = schedule["segments"][0]["start_lr"]
        schedule["shape"] = "piecewise"
        schedule["shape_pattern"] = shape_pattern(schedule)
        schedule["name"] = format_muon_schedule(schedule)
    return schedules


def segment_lr_at_step(segment, step):
    span = max(segment["end_step"] - segment["start_step"], 1)
    x = min(max((step - segment["start_step"]) / span, 0.0), 1.0)
    start_lr = segment["start_lr"]
    end_lr = segment["end_lr"]
    if segment["shape"] == "linear":
        return start_lr + (end_lr - start_lr) * x
    if segment["shape"] == "power":
        return end_lr + (start_lr - end_lr) * ((1 - x) ** segment["power"])
    if segment["shape"] == "exp":
        safe_start = max(start_lr, MIN_EXP_LR)
        safe_end = max(end_lr, MIN_EXP_LR)
        return safe_start * ((safe_end / safe_start) ** x)
    raise ValueError("Unknown segment shape: %s" % segment["shape"])


def muon_lr_at_step(schedule, step, total_train_steps):
    del total_train_steps
    for segment in schedule["segments"]:
        if segment["start_step"] <= step < segment["end_step"]:
            return segment_lr_at_step(segment, step)
    return segment_lr_at_step(schedule["segments"][-1], step)


def print_sweep_summary(sweep_results):
    ranked = sorted(
        sweep_results,
        key=lambda result: (result["tta_val_acc"], result["val_acc"], -result["train_loss"]),
        reverse=True,
    )
    tta_accs = torch.tensor([result["tta_val_acc"] for result in sweep_results])
    val_accs = torch.tensor([result["val_acc"] for result in sweep_results])
    print("\nMuon LR scheduler sweep summary")
    print("schedules: %d" % len(sweep_results))
    print("tta mean: %.4f    tta std: %.4f    best: %.4f    worst: %.4f" % (
        tta_accs.mean().item(), tta_accs.std().item(), tta_accs.max().item(), tta_accs.min().item()
    ))
    print("val mean: %.4f    val std: %.4f    best: %.4f    worst: %.4f" % (
        val_accs.mean().item(), val_accs.std().item(), val_accs.max().item(), val_accs.min().item()
    ))
    top_patterns = sorted(
        {result["shape_pattern"] for result in sweep_results},
        key=lambda pattern: max(
            result["tta_val_acc"] for result in sweep_results
            if result["shape_pattern"] == pattern
        ),
        reverse=True,
    )[:10]
    for pattern in top_patterns:
        pattern_results = [result for result in sweep_results if result["shape_pattern"] == pattern]
        pattern_tta = torch.tensor([result["tta_val_acc"] for result in pattern_results])
        best = max(pattern_results, key=lambda result: (result["tta_val_acc"], result["val_acc"]))
        print(
            "%-4s count=%2d mean_tta=%.4f best_tta=%.4f best=%s"
            % (pattern, len(pattern_results), pattern_tta.mean().item(), best["tta_val_acc"], best["schedule"])
        )

    print("\nMuon LR scheduler ranking")
    print("rank | run | patt | val_acc | tta_val_acc | train_loss | schedule")
    print("-" * 140)
    for rank, result in enumerate(ranked, start=1):
        print(
            "%4d | %3d | %-4s | %.4f  | %.4f      | %.4f     | %s"
            % (
                rank,
                result["index"],
                result["shape_pattern"],
                result["val_acc"],
                result["tta_val_acc"],
                result["train_loss"],
                result["schedule"],
            )
        )
    return ranked


def main(run, model, muon_schedule):
    set_training_seed()

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
    optimizer2 = Muon(filter_params, lr=muon_schedule["initial_lr"], momentum=0.6, nesterov=True)
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
        for inputs, labels in train_loader:
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            loss = F.cross_entropy(outputs, labels, label_smoothing=0.2, reduction="sum")
            loss.backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:]:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for group in optimizer2.param_groups:
                group["lr"] = muon_lr_at_step(muon_schedule, step, total_train_steps)
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
        train_loss = F.cross_entropy(outputs.detach().float(), labels, label_smoothing=0.2).item()
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    return dict(train_loss=train_loss, train_acc=train_acc, val_acc=val_acc, tta_val_acc=tta_val_acc)

if __name__ == "__main__":

    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    print_columns(logging_columns_list, is_head=True)
    # main("warmup", model)
    sweep_results = []
    for schedule in make_muon_schedules():
        run = "%03d %s" % (schedule["index"], schedule["name"])
        print("\nMuon schedule %s" % run, flush=True)
        result = main(run, model, schedule)
        result.update(
            index=schedule["index"],
            schedule=schedule["name"],
            shape=schedule["shape"],
            shape_pattern=schedule["shape_pattern"],
            initial_lr=schedule["initial_lr"],
            segments=schedule["segments"],
            profile=schedule["profile"],
        )
        sweep_results.append(result)
        print("Train loss: %.4f" % result["train_loss"])
        print("Train acc:  %.4f" % result["train_acc"])
        print("Val acc:    %.4f" % result["val_acc"])
        print("TTA val:    %.4f" % result["tta_val_acc"])

    ranking = print_sweep_summary(sweep_results)

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, sweep_results=sweep_results, ranking=ranking), log_path)
    print(os.path.abspath(log_path))
