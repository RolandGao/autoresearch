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
import gc

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil

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
USE_CUDA_GRAPHS = True
CUDA_GRAPH_BATCH_SIZES = {125, 500, 2000}
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
    where S' is diagonal with S_{ii}' \\sim Uniform(0.5, 1.5), which turns out not to hurt model
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
        if not torch.is_tensor(lr) and lr < 0.0:
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
                if torch.is_tensor(lr):
                    p.data.addcmul_(update, lr, value=-1)  # take a step
                else:
                    p.data.add_(update, alpha=-lr)  # take a step


class CUDAGraphTrainer:
    def __init__(self, model, batch_size, sgd_groups, muon_params):
        self.model = model
        self.batch_size = batch_size
        self.static_inputs = torch.empty(
            batch_size, 3, 32, 32, device="cuda", dtype=torch.float16
        ).to(memory_format=torch.channels_last)
        self.static_labels = torch.empty(batch_size, device="cuda", dtype=torch.long)
        self.static_inputs.zero_()
        self.static_labels.zero_()
        self.sgd_lr_tensors = [
            torch.zeros((), device="cuda", dtype=torch.float32) for _ in sgd_groups
        ]
        self.muon_lr_tensor = torch.zeros((), device="cuda", dtype=torch.float32)
        whiten_param_config = [
            dict(
                params=sgd_groups[0]["params"],
                lr=self.sgd_lr_tensors[0],
                weight_decay=sgd_groups[0]["weight_decay"],
            )
        ]
        main_param_configs = [
            dict(
                params=group["params"],
                lr=self.sgd_lr_tensors[group_index],
                weight_decay=group["weight_decay"],
            )
            for group_index, group in enumerate(sgd_groups[1:], start=1)
        ]
        self.optimizer_whiten = torch.optim.SGD(
            whiten_param_config, momentum=0.85, nesterov=True, fused=True
        )
        self.optimizer_main = torch.optim.SGD(
            main_param_configs, momentum=0.85, nesterov=True, fused=True
        )
        self.optimizer2 = Muon(
            muon_params, lr=self.muon_lr_tensor, momentum=0.6, nesterov=True
        )
        self.optimizers = [self.optimizer_whiten, self.optimizer_main, self.optimizer2]
        self.graphs = {}
        self.outputs = {}
        self._initialize_optimizer_state()
        self._capture_graphs()

    def _model_tensors(self):
        tensors = []
        tensors.extend(param.data for param in self.model.parameters())
        tensors.extend(buffer.data for buffer in self.model.buffers())
        return tensors

    def _optimizer_state_tensors(self):
        tensors = []
        for opt in self.optimizers:
            for state in opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        tensors.append(value)
        return tensors

    def _state_tensors(self):
        return self._model_tensors() + self._optimizer_state_tensors()

    def _snapshot_state(self):
        return [tensor.detach().clone() for tensor in self._state_tensors()]

    def _restore_state(self, snapshot):
        for tensor, value in zip(self._state_tensors(), snapshot):
            tensor.copy_(value)
        self._zero_grads()

    def reset_optimizer_state(self):
        for tensor in self._optimizer_state_tensors():
            tensor.zero_()
        self._zero_grads()

    def _zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=False)

    def _fill_lrs(self, sgd_lrs, muon_lr):
        for lr_tensor, lr in zip(self.sgd_lr_tensors, sgd_lrs):
            lr_tensor.fill_(lr)
        self.muon_lr_tensor.fill_(muon_lr)

    def _run_step(self, whiten_bias_grad):
        outputs = self.model(self.static_inputs, whiten_bias_grad=whiten_bias_grad)
        loss = F.cross_entropy(
            outputs, self.static_labels, label_smoothing=0.2, reduction="sum"
        )
        loss.backward()
        if whiten_bias_grad:
            self.optimizer_whiten.step()
        self.optimizer_main.step()
        self.optimizer2.step()
        self._zero_grads()
        return outputs

    def _initialize_optimizer_state(self):
        model_snapshot = [tensor.detach().clone() for tensor in self._model_tensors()]
        self._fill_lrs((0.0, 0.0, 0.0), 0.0)
        self.model.train()
        self._run_step(True)
        self._run_step(False)
        for tensor, value in zip(self._model_tensors(), model_snapshot):
            tensor.copy_(value)
        self.reset_optimizer_state()

    def _capture_one_graph(self, whiten_bias_grad, snapshot):
        self._restore_state(snapshot)
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._run_step(whiten_bias_grad)
        torch.cuda.current_stream().wait_stream(side_stream)

        self._restore_state(snapshot)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.outputs[whiten_bias_grad] = self._run_step(whiten_bias_grad)
        self.graphs[whiten_bias_grad] = graph
        self._restore_state(snapshot)

    def _capture_graphs(self):
        snapshot = self._snapshot_state()
        self._capture_one_graph(True, snapshot)
        self._capture_one_graph(False, snapshot)

    def step(self, inputs, labels, sgd_lrs, muon_lr, whiten_bias_grad):
        self.static_inputs.copy_(inputs)
        self.static_labels.copy_(labels)
        self._fill_lrs(sgd_lrs, muon_lr)
        self.graphs[whiten_bias_grad].replay()
        return self.outputs[whiten_bias_grad]


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
    def __init__(
        self, path, train=True, batch_size=500, aug=None, shuffle=None, drop_last=None
    ):
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
        self.drop_last = train if drop_last is None else drop_last
        self.shuffle = train if shuffle is None else shuffle

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
    "run   ",
    "epoch",
    "train_loss",
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


def evaluate_train_loss(model, batches):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for inputs, labels in batches:
            outputs = model(inputs)
            total_loss += F.cross_entropy(
                outputs.float(), labels, label_smoothing=0.2, reduction="sum"
            ).item()
            total_examples += len(labels)
    return total_loss / total_examples


def calibrate_batchnorm(model, batches):
    batchnorms = [module for module in model.modules() if isinstance(module, BatchNorm)]
    momentums = [module.momentum for module in batchnorms]
    for module in batchnorms:
        module.reset_running_stats()
        module.momentum = None

    model.train()
    with torch.inference_mode():
        for inputs, _ in batches:
            model(inputs)

    model.eval()
    for module, momentum in zip(batchnorms, momentums):
        module.momentum = momentum


def make_train_eval_batches():
    loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=BN_CAL_BATCH_SIZE,
        aug=dict(flip=True, translate=2),
        shuffle=False,
        drop_last=False,
    )
    batches = []
    for inputs, labels in loader:
        batches.append((inputs.detach(), labels.detach()))
        if len(batches) >= BN_CAL_BATCHES:
            break
    return batches


############################################
#                Training                  #
############################################


def log_linear_values(start, end, count):
    if count == 1:
        return [start]
    ratio = end / start
    return [start * ratio ** (index / (count - 1)) for index in range(count)]


MUON_LR_VALUES = log_linear_values(0.1, 5.0, 20)
BATCH_SIZE_VALUES = (2000, 10000, 50000)
BN_CAL_BATCH_SIZE = 2000
BN_CAL_BATCHES = 25


def make_muon_schedules():
    schedules = []
    sanity_configs = [(2000, 0.24), (50000, 0.24)]
    configs = [
        (batch_size, lr) for batch_size in BATCH_SIZE_VALUES for lr in MUON_LR_VALUES
    ]
    configs = sanity_configs + [
        config
        for config in configs
        if not any(
            config[0] == sanity_config[0]
            and abs(config[1] - sanity_config[1]) < 1e-12
            for sanity_config in sanity_configs
        )
    ]
    for batch_size, lr in configs:
        schedules.append(
            dict(
                index=len(schedules) + 1,
                batch_size=batch_size,
                micro_batch_size=batch_size,
                initial_lr=lr,
                shape="linear_decay",
                shape_pattern="L",
                name="bs%d linear %.2g->0" % (batch_size, lr),
                profile="batch_size_2000_10000_50000_linear_decay_muon_lr_0.1_to_5_loglinear_20",
                segments=[dict(shape="linear", start_lr=lr, end_lr=0.0)],
            )
        )
    return schedules


def divisible_batch_sizes(batch_size):
    return [
        candidate
        for candidate in range(batch_size, 0, -1)
        if batch_size % candidate == 0
    ]


def is_cuda_oom(error):
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def cleanup_after_cuda_oom(graph_trainers):
    graph_trainers.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def muon_lr_at_step(schedule, step, total_train_steps):
    progress = min(max(step / total_train_steps, 0.0), 1.0)
    return schedule["initial_lr"] * (1 - progress)


def print_sweep_summary(sweep_results):
    ranked = sorted(
        sweep_results,
        key=lambda result: (
            result["tta_val_acc"],
            result["val_acc"],
            -result["train_loss"],
        ),
        reverse=True,
    )
    tta_accs = torch.tensor([result["tta_val_acc"] for result in sweep_results])
    val_accs = torch.tensor([result["val_acc"] for result in sweep_results])
    print("\nMuon LR scheduler sweep summary")
    print("schedules: %d" % len(sweep_results))
    print(
        "tta mean: %.4f    tta std: %.4f    best: %.4f    worst: %.4f"
        % (
            tta_accs.mean().item(),
            tta_accs.std().item(),
            tta_accs.max().item(),
            tta_accs.min().item(),
        )
    )
    print(
        "val mean: %.4f    val std: %.4f    best: %.4f    worst: %.4f"
        % (
            val_accs.mean().item(),
            val_accs.std().item(),
            val_accs.max().item(),
            val_accs.min().item(),
        )
    )
    top_patterns = sorted(
        {result["shape_pattern"] for result in sweep_results},
        key=lambda pattern: max(
            result["tta_val_acc"]
            for result in sweep_results
            if result["shape_pattern"] == pattern
        ),
        reverse=True,
    )[:10]
    for pattern in top_patterns:
        pattern_results = [
            result for result in sweep_results if result["shape_pattern"] == pattern
        ]
        pattern_tta = torch.tensor(
            [result["tta_val_acc"] for result in pattern_results]
        )
        best = max(
            pattern_results,
            key=lambda result: (result["tta_val_acc"], result["val_acc"]),
        )
        print(
            "%-4s count=%2d mean_tta=%.4f best_tta=%.4f best=%s"
            % (
                pattern,
                len(pattern_results),
                pattern_tta.mean().item(),
                best["tta_val_acc"],
                best["schedule"],
            )
        )

    print("\nMuon LR scheduler ranking")
    print(
        "rank | run | bs | micro | accum | patt | 25batch_loss | val_acc | tta_val_acc | bn_cal_25batch_loss | bn_cal_val | bn_cal_tta | schedule"
    )
    print("-" * 190)
    for rank, result in enumerate(ranked, start=1):
        print(
            "%4d | %3d | %5d | %5d | %5d | %-4s | %.4f       | %.4f  | %.4f      | %.4f             | %.4f     | %.4f     | %s"
            % (
                rank,
                result["index"],
                result["batch_size"],
                result["micro_batch_size"],
                result["accum_steps"],
                result["shape_pattern"],
                result["train25_loss"],
                result["val_acc"],
                result["tta_val_acc"],
                result["bn_cal_train25_loss"],
                result["bn_cal_val_acc"],
                result["bn_cal_tta_val_acc"],
                result["schedule"],
            )
        )
    return ranked


def main(run, model, muon_schedule, graph_trainers):
    set_training_seed()

    batch_size = muon_schedule["batch_size"]
    micro_batch_size = muon_schedule.get("micro_batch_size", batch_size)
    assert batch_size % micro_batch_size == 0
    accum_steps = batch_size // micro_batch_size
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader(
        "cifar10",
        train=True,
        batch_size=micro_batch_size,
        aug=dict(flip=True, translate=2),
    )
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
    train_steps_per_epoch = len(train_loader) // accum_steps
    total_train_steps = ceil(8 * train_steps_per_epoch)
    whiten_bias_train_steps = ceil(3 * train_steps_per_epoch)

    # Create optimizers and learning rate schedulers
    filter_params = [
        p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad
    ]
    norm_biases = [
        p for n, p in model.named_parameters() if "norm" in n and p.requires_grad
    ]
    sgd_groups = [
        dict(params=[model.whiten.bias], initial_lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=norm_biases, initial_lr=bias_lr, weight_decay=wd / bias_lr),
        dict(params=[model.head.weight], initial_lr=head_lr, weight_decay=wd / head_lr),
    ]
    use_cuda_graph = (
        USE_CUDA_GRAPHS
        and accum_steps == 1
        and micro_batch_size in CUDA_GRAPH_BATCH_SIZES
    )
    if not use_cuda_graph:
        param_configs = [
            dict(
                params=group["params"],
                lr=group["initial_lr"],
                weight_decay=group["weight_decay"],
            )
            for group in sgd_groups
        ]
        optimizer1 = torch.optim.SGD(
            param_configs, momentum=0.85, nesterov=True, fused=True
        )
        optimizer2 = Muon(
            filter_params, lr=muon_schedule["initial_lr"], momentum=0.6, nesterov=True
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
    train_eval_batches = []

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalized_images()[:5000]
    model.init_whiten(train_images)
    if use_cuda_graph:
        if micro_batch_size not in graph_trainers:
            graph_trainers[micro_batch_size] = CUDAGraphTrainer(
                model, micro_batch_size, sgd_groups, filter_params
            )
        trainer = graph_trainers[micro_batch_size]
        trainer.reset_optimizer_state()
    else:
        trainer = None
    stop_timer()

    for epoch in range(ceil(total_train_steps / train_steps_per_epoch)):
        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        accum_outputs = []
        accum_labels = []
        micro_step = 0
        for inputs, labels in train_loader:
            if micro_batch_size == BN_CAL_BATCH_SIZE:
                train_eval_batches.append((inputs.detach(), labels.detach()))
                train_eval_batches = train_eval_batches[-BN_CAL_BATCHES:]
            if use_cuda_graph:
                whiten_bias_grad = step < whiten_bias_train_steps
                whiten_lr = bias_lr * (1 - step / whiten_bias_train_steps)
                main_lr_factor = 1 - step / total_train_steps
                sgd_lrs = (
                    whiten_lr,
                    bias_lr * main_lr_factor,
                    head_lr * main_lr_factor,
                )
                outputs = trainer.step(
                    inputs,
                    labels,
                    sgd_lrs,
                    muon_lr_at_step(muon_schedule, step, total_train_steps),
                    whiten_bias_grad,
                )
            else:
                if micro_step == 0:
                    model.zero_grad(set_to_none=True)
                outputs = model(
                    inputs, whiten_bias_grad=(step < whiten_bias_train_steps)
                )
                loss = F.cross_entropy(
                    outputs, labels, label_smoothing=0.2, reduction="sum"
                )
                loss.backward()
                accum_outputs.append(outputs.detach())
                accum_labels.append(labels.detach())
                micro_step += 1
                if micro_step < accum_steps:
                    continue

                for group in optimizer1.param_groups[:1]:
                    group["lr"] = group["initial_lr"] * (
                        1 - step / whiten_bias_train_steps
                    )
                for group in optimizer1.param_groups[1:]:
                    group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
                for group in optimizer2.param_groups:
                    group["lr"] = muon_lr_at_step(
                        muon_schedule, step, total_train_steps
                    )
                for opt in optimizers:
                    opt.step()
                model.zero_grad(set_to_none=True)
                outputs = torch.cat(accum_outputs)
                labels = torch.cat(accum_labels)
                accum_outputs = []
                accum_labels = []
                micro_step = 0
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
        run = None  # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    if micro_batch_size != BN_CAL_BATCH_SIZE:
        set_training_seed()
        train_eval_batches = make_train_eval_batches()
    train25_loss = evaluate_train_loss(model, train_eval_batches)
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    start_timer()
    calibrate_batchnorm(model, train_eval_batches)
    bn_cal_train25_loss = evaluate_train_loss(model, train_eval_batches)
    bn_cal_val_acc = evaluate(model, test_loader, tta_level=0)
    bn_cal_tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()

    return {
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "accum_steps": accum_steps,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train25_loss": train25_loss,
        "25batch_train_loss": train25_loss,
        "val_acc": val_acc,
        "tta_val_acc": tta_val_acc,
        "bn_cal_train25_loss": bn_cal_train25_loss,
        "bn_cal_25batch_train_loss": bn_cal_train25_loss,
        "bn_cal_val_acc": bn_cal_val_acc,
        "bn_cal_tta_val_acc": bn_cal_tta_val_acc,
    }


if __name__ == "__main__":
    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    set_training_seed()
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    # model.compile(mode="max-autotune")

    print_columns(logging_columns_list, is_head=True)
    # main("warmup", model)
    sweep_results = []
    graph_trainers = {}
    micro_batch_size_cache = {}
    for schedule in make_muon_schedules():
        batch_size = schedule["batch_size"]
        candidate_micro_batch_sizes = (
            [micro_batch_size_cache[batch_size]]
            if batch_size in micro_batch_size_cache
            else divisible_batch_sizes(batch_size)
        )
        result = None
        schedule_to_run = None
        for micro_batch_size in candidate_micro_batch_sizes:
            schedule_to_run = dict(schedule, micro_batch_size=micro_batch_size)
            accum_steps = batch_size // micro_batch_size
            run = "%03d %s micro%d accum%d" % (
                schedule["index"],
                schedule["name"],
                micro_batch_size,
                accum_steps,
            )
            print("\nMuon schedule %s" % run, flush=True)
            try:
                result = main(run, model, schedule_to_run, graph_trainers)
            except RuntimeError as error:
                if not is_cuda_oom(error):
                    raise
                print(
                    "CUDA OOM with effective batch %d, microbatch %d; trying the next divisor."
                    % (batch_size, micro_batch_size),
                    flush=True,
                )
                cleanup_after_cuda_oom(graph_trainers)
                result = None
                continue

            micro_batch_size_cache.setdefault(batch_size, micro_batch_size)
            if accum_steps > 1:
                print(
                    "Using gradient accumulation: effective batch %d = %d x microbatch %d"
                    % (batch_size, accum_steps, micro_batch_size),
                    flush=True,
                )
            break

        if result is None:
            raise RuntimeError(
                "All divisible microbatch sizes OOM for effective batch %d" % batch_size
            )

        result.update(
            index=schedule["index"],
            schedule=schedule["name"],
            shape=schedule["shape"],
            shape_pattern=schedule["shape_pattern"],
            batch_size=schedule_to_run["batch_size"],
            micro_batch_size=schedule_to_run["micro_batch_size"],
            accum_steps=schedule_to_run["batch_size"]
            // schedule_to_run["micro_batch_size"],
            initial_lr=schedule_to_run["initial_lr"],
            segments=schedule_to_run["segments"],
            profile=schedule_to_run["profile"],
        )
        sweep_results.append(result)
        print("Batch size:  %d" % result["batch_size"])
        print("Microbatch:  %d" % result["micro_batch_size"])
        print("Accum steps: %d" % result["accum_steps"])
        print("Train loss: %.4f" % result["train_loss"])
        print("Train acc:  %.4f" % result["train_acc"])
        print("25batch train loss:        %.4f" % result["train25_loss"])
        print("Val acc:                   %.4f" % result["val_acc"])
        print("TTA val:                   %.4f" % result["tta_val_acc"])
        print("BN cal 25batch train loss: %.4f" % result["bn_cal_train25_loss"])
        print("BN cal val acc:            %.4f" % result["bn_cal_val_acc"])
        print("BN cal TTA val:            %.4f" % result["bn_cal_tta_val_acc"])

    ranking = print_sweep_summary(sweep_results)

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, sweep_results=sweep_results, ranking=ranking), log_path)
    print(os.path.abspath(log_path))
