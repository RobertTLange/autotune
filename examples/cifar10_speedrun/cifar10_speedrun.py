#############################################
#                  Setup                    #
#############################################

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from functools import wraps
from math import ceil
from typing import Callable

import numpy as np
import scipy.stats
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

ACCURACY_THRESHOLD = 0.94
DEFAULT_NUM_RUNS = 5
DEFAULT_MAX_TIME_PER_RUN = 3.0
DEFAULT_DATA_DIR = "cifar10"
TTA_UNCERTAIN_QUANTILE = 0.25


@dataclass(frozen=True)
class SpeedrunConfig:
    training_batch_size: int
    bias_lr: float
    head_lr: float
    weight_decay_scale: float
    sgd_momentum: float
    muon_lr: float
    muon_momentum: float
    train_epochs: float
    whiten_bias_epochs: float
    label_smoothing: float
    translate: int
    brightness_range: float
    contrast_range: float
    data_dir: str


class CustomTorchCompile:
    """
    A decorator class for torch.compile that compiles a function on its second and subsequent calls.

    On the first call, runs the original (interpreted) function and sets up the compiled version;
    on further calls, uses the compiled function for efficiency. This two-phase execution is useful
    for coverage tracking with the `coverage` library: coverage tools only see the first (interpreted)
    execution, ensuring accurate source-level coverage data even for code that will ultimately run
    in the compiled mode.

    Usage:
        @CustomTorchCompile(...)
        def some_fn(...):
            ...

    Args:
        *compile_args: Positional arguments for torch.compile.
        **compile_kwargs: Keyword arguments for torch.compile.
    """

    def __init__(self, *compile_args, **compile_kwargs) -> None:
        self._compile_args = compile_args
        self._compile_kwargs = compile_kwargs
        self._compiled_func = None
        self._first_call = True

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._first_call:
                self._first_call = False
                self._compiled_func = torch.compile(
                    *self._compile_args, **self._compile_kwargs
                )(func)
                return func(*args, **kwargs)
            else:
                return self._compiled_func(*args, **kwargs)

        return wrapper


#############################################
#               Muon optimizer              #
#############################################


@CustomTorchCompile(fullgraph=True)
def _zeropower_via_newtonschulz5(
    gradients_4d: list[torch.half],
    filter_meta_data: list[tuple],
    max_D: int,
    max_K: int,
    current_step: int,
    total_steps: int,
) -> list[torch.half]:
    a, b, c = (3.4576, -4.7391, 2.0843)
    eps_stable = 1e-05
    eps_gms = 1e-05
    progress_ratio = current_step / max(1, total_steps)

    initial_target_mag = 0.5012
    final_target_mag = 0.0786
    target_magnitude = (
        initial_target_mag * (1 - progress_ratio) + final_target_mag * progress_ratio
    )

    # Use stack instead of pre-allocated tensor for better performance
    if not filter_meta_data:
        return gradients_4d

    grad_list = []
    for meta in filter_meta_data:
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        grad_to_orthogonalize = gradients_4d[list_idx]
        g_reshaped = grad_to_orthogonalize.reshape(reshaped_D, reshaped_K)
        padding_dims = (0, max_K - reshaped_K, 0, max_D - reshaped_D)
        g_padded = F.pad(g_reshaped, padding_dims, "constant", 0)
        grad_list.append(g_padded)

    if not grad_list:
        return gradients_4d

    X = torch.stack(grad_list)

    # Fuse normalization operations for better performance
    current_batch_mags = X.norm(dim=(1, 2), keepdim=True)
    scale_factor = target_magnitude / (current_batch_mags + eps_gms)
    X = X * scale_factor

    X_norm = X.norm(dim=(1, 2), keepdim=True)
    X = X / (X_norm + eps_stable)

    transposed = False
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
        transposed = True

    # Unroll the loop for better performance
    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X

    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X

    A = X @ X.transpose(1, 2)
    B = b * A + c * (A @ A)
    X = a * X + B @ X

    if transposed:
        X = X.transpose(1, 2)

    final_orthogonalized_grads_list = [None] * len(gradients_4d)
    for i, meta in enumerate(filter_meta_data):
        original_shape, reshaped_D, reshaped_K, list_idx = meta
        orthogonalized_g_padded = X[i]
        orthogonalized_g_reshaped = orthogonalized_g_padded[:reshaped_D, :reshaped_K]
        final_orthogonalized_grads_list[list_idx] = orthogonalized_g_reshaped.view(
            original_shape
        )
    return final_orthogonalized_grads_list


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.08,
        momentum=0.88,
        nesterov=True,
        norm_freq=1,
        total_train_steps=None,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            norm_freq=norm_freq,
            total_train_steps=total_train_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.step_count = 0
        self.last_norm_step = 0
        self.total_train_steps = total_train_steps
        self.filter_params_meta = []
        self.max_D, self.max_K = (0, 0)
        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 4 and p.requires_grad:
                    reshaped_D = p.shape[0]
                    reshaped_K = p.data.numel() // p.shape[0]
                    self.filter_params_meta.append(
                        {
                            "param": p,
                            "original_shape": p.data.shape,
                            "reshaped_dims": (reshaped_D, reshaped_K),
                        }
                    )
                    self.max_D = max(self.max_D, reshaped_D)
                    self.max_K = max(self.max_K, reshaped_K)
        self.max_D = max(1, self.max_D)
        self.max_K = (max(1, self.max_K) + 15) // 16 * 16
        self.current_grad_norms = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        group = self.param_groups[0]
        progress = self.step_count / self.total_train_steps
        group["norm_freq"] = 2 + int(15 * progress)
        # Prepare momentum buffers and track meta data
        filter_params_with_grad = []
        filter_meta_for_current_step = []
        momentum_buffers = [] if group["momentum_buffer_dtype"] == torch.half else None

        for p_meta in self.filter_params_meta:
            p = p_meta["param"]
            if p.grad is not None:
                filter_params_with_grad.append(p)
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        p.grad,
                        dtype=group["momentum_buffer_dtype"],
                        memory_format=torch.preserve_format,
                    )
                if momentum_buffers is not None:
                    momentum_buffers.append(state["momentum_buffer"])
                filter_meta_for_current_step.append(
                    (
                        p_meta["original_shape"],
                        p_meta["reshaped_dims"][0],
                        p_meta["reshaped_dims"][1],
                        len(filter_params_with_grad)
                        - 1,  # Index in filter_params_with_grad
                    )
                )

        if not filter_params_with_grad:
            return

        # Apply momentum and add gradients
        if momentum_buffers is not None:
            torch._foreach_mul_(momentum_buffers, group["momentum"])
            grad_casts = [
                g.to(mb.dtype)
                for g, mb in zip(
                    [p.grad for p in filter_params_with_grad], momentum_buffers
                )
            ]
            torch._foreach_add_(momentum_buffers, grad_casts)
        else:
            momentum_buffers = [p.grad for p in filter_params_with_grad]

        if group["nesterov"]:
            nesterov_grads = torch._foreach_add(
                [p.grad for p in filter_params_with_grad],
                momentum_buffers,
                alpha=group["momentum"],
            )
        else:
            nesterov_grads = momentum_buffers

        do_norm_scaling = self.step_count - self.last_norm_step >= group["norm_freq"]
        if do_norm_scaling:
            self.last_norm_step = self.step_count
            self.current_grad_norms = torch._foreach_norm(filter_params_with_grad)
            scale_factors = [
                (len(p.data) ** 0.5 / (n + 1e-07)).to(p.data.dtype)
                for p, n in zip(filter_params_with_grad, self.current_grad_norms)
            ]

        final_orthogonalized_grads = _zeropower_via_newtonschulz5(
            nesterov_grads,
            filter_meta_for_current_step,
            self.max_D,
            self.max_K,
            self.step_count,
            self.total_train_steps,
        )

        # Apply updates in a single fused operation when possible
        if do_norm_scaling:
            # Scale gradients first
            torch._foreach_mul_(filter_params_with_grad, scale_factors)
            # Then apply the orthogonalized updates
            torch._foreach_add_(
                filter_params_with_grad, final_orthogonalized_grads, alpha=-group["lr"]
            )
        else:
            # Apply optimizer step directly
            torch._foreach_add_(
                filter_params_with_grad, final_orthogonalized_grads, alpha=-group["lr"]
            )

        # Apply weight decay in a fused operation
        weight_decay_factor = 1 - group["lr"] * group["weight_decay"]
        if weight_decay_factor != 1.0:
            torch._foreach_mul_(filter_params_with_grad, weight_decay_factor)

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()


#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.half)
CIFAR_STD = torch.tensor((0.247, 0.2435, 0.2616), dtype=torch.half)


@CustomTorchCompile()
def batch_color_jitter(inputs, brightness_range: float, contrast_range: float):
    B = inputs.shape[0]
    device = inputs.device
    dtype = inputs.dtype
    brightness_shift = (
        torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
    ) * brightness_range
    contrast_scale = (
        torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 2 - 1
    ) * contrast_range + 1
    inputs = inputs + brightness_shift
    inputs = inputs * contrast_scale
    return inputs


@CustomTorchCompile()
def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


@CustomTorchCompile()
def batch_crop(images, crop_size):
    B, C, H_padded, W_padded = images.shape
    r = (H_padded - crop_size) // 2
    y_offsets = (torch.rand(B, device=images.device) * (2 * r + 1)).long()
    x_offsets = (torch.rand(B, device=images.device) * (2 * r + 1)).long()
    base_y_coords = torch.arange(crop_size, device=images.device).view(
        1, 1, crop_size, 1
    )
    base_x_coords = torch.arange(crop_size, device=images.device).view(
        1, 1, 1, crop_size
    )
    y_start_coords_expanded = y_offsets.view(B, 1, 1, 1)
    x_start_coords_expanded = x_offsets.view(B, 1, 1, 1)
    y_indices = y_start_coords_expanded + base_y_coords
    y_indices = y_indices.expand(B, C, crop_size, crop_size)
    x_indices = x_start_coords_expanded + base_x_coords
    x_indices = x_indices.expand(B, C, crop_size, crop_size)
    batch_indices = (
        torch.arange(B, device=images.device).view(B, 1, 1, 1).expand_as(y_indices)
    )
    channel_indices = (
        torch.arange(C, device=images.device).view(1, C, 1, 1).expand_as(y_indices)
    )
    cropped_images = images[batch_indices, channel_indices, y_indices, x_indices]
    return cropped_images


class CifarLoader:
    def __init__(self, path, train=True, batch_size=500, aug=None):
        if path == "cifar10":
            path = os.path.expanduser("~/data/cifar10")
        data_path = os.path.join(path, "train.pt" if train else "test.pt")
        if not os.path.exists(data_path):
            print(f"Downloading CIFAR-10 {'train' if train else 'test'} data to {path}...")
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save(
                {"images": images, "labels": labels, "classes": dset.classes}, data_path
            )
        else:
            print(f"Using CIFAR-10 data from {data_path}")
        data = torch.load(
            data_path, map_location=torch.device("cuda"), weights_only=True
        )
        self.images, self.labels, self.classes = (
            data["images"],
            data["labels"],
            data["classes"],
        )
        self.images = (
            (self.images.half() / 255)
            .permute(0, 3, 1, 2)
            .to(memory_format=torch.channels_last)
        )
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {}
        self.epoch = 0
        self.aug = aug or {}
        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train
        # Pre-allocate indices tensor for better performance
        self._indices = torch.empty(len(self.images), dtype=torch.long, device="cuda")

    def __len__(self):
        return (
            len(self.images) // self.batch_size
            if self.drop_last
            else ceil(len(self.images) / self.batch_size)
        )

    def __iter__(self):
        if self.epoch == 0:
            images = self.proc_images["norm"] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get("flip", False):
                images = self.proc_images["flip"] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get("translate", 0)
            if pad > 0:
                self.proc_images["pad"] = F.pad(images, (pad,) * 4, "reflect")

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

        color_jitter_config = self.aug.get("color_jitter", {"enabled": False})
        if color_jitter_config.get("enabled", False):
            brightness = color_jitter_config.get("brightness_range", 0.1)
            contrast = color_jitter_config.get("contrast_range", 0.1)
            images = batch_color_jitter(images, brightness, contrast)

        self.epoch += 1

        if self.shuffle:
            torch.randperm(len(self._indices), out=self._indices)
            indices = self._indices
        else:
            indices = torch.arange(len(self.images), device=self.images.device)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield (images[idxs], self.labels[idxs])


#############################################
#            Network Definition             #
#############################################


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.5566, eps=1e-12):
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
        self.activ = nn.SiLU()

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
            mod.half()
        self.to(memory_format=torch.channels_last)

    def reset(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        w = self.head.weight.data
        w.mul_(1.0 / w.std())

    def init_whiten(self, train_images, eps=0.0005):
        c, (h, w) = (train_images.shape[1], self.whiten.weight.shape[2:])
        patches = (
            train_images.unfold(2, h, 1)
            .unfold(3, w, 1)
            .transpose(1, 3)
            .reshape(-1, c, h, w)
            .float()
        )
        patches_flat = patches.view(len(patches), -1)
        # Use more efficient covariance computation with SVD for better numerical stability
        est_patch_covariance = torch.mm(patches_flat.t(), patches_flat) / len(
            patches_flat
        )
        U, S, V = torch.svd(est_patch_covariance)
        # More stable inverse square root computation
        inv_sqrt_S = torch.rsqrt(S + eps)
        eigenvectors_scaled = (U * inv_sqrt_S.unsqueeze(0)).T.reshape(-1, c, h, w)
        self.whiten.weight.data[:] = torch.cat(
            (eigenvectors_scaled, -eigenvectors_scaled)
        )

    def forward(self, x, whiten_bias_grad=True):
        x = x.to(memory_format=torch.channels_last)
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1).contiguous()
        return self.head(x) / x.size(-1)


############################################
#           Logging & Metrics              #
############################################


def compute_logs(
    model: nn.Module,
    val_acc: float,
    train_acc: float,
    time_seconds: float,
    **extra_info,
) -> dict[str, float]:
    """
    Compute logs for downstream analysis and hypothesis evaluation.

    This function is the PRIMARY EXTENSION POINT for adding custom metrics.
    Add your metrics here so they are properly logged and available for
    hypothesis testing.

    Args:
        model: The trained model (can access weights for analysis, e.g.,
               orthogonality checks, condition numbers, etc.)
        val_acc: Validation accuracy for this epoch
        train_acc: Training accuracy for this epoch
        time_seconds: Elapsed training time in seconds
        **extra_info: Any additional keyword arguments to include in logs

    Returns:
        Dict of metrics that will be used for hypothesis evaluation.
        All values should be floats.

    Example - Adding custom metrics:
        # Inside this function, you can add:

        # 1. Weight-based metrics
        # logs["weight_norm"] = float(model.head.weight.norm().item())

        # 2. Covariance/isotropy metrics
        # logs["condition_number"] = compute_condition_number(some_matrix)
        # logs["off_diagonal_energy"] = compute_off_diagonal_energy(cov_matrix)

        # 3. Any hypothesis-specific measurements
        # logs["my_custom_metric"] = compute_my_metric(model)
    """
    # Base metrics (always included)
    logs = {
        "val_acc": float(val_acc),
        "train_acc": float(train_acc),
        "time_seconds": float(time_seconds),
    }

    # Add any extra info passed as kwargs
    for key, value in extra_info.items():
        if value is not None:
            logs[key] = float(value) if isinstance(value, (int, float)) else value

    # ============================================================
    # TODO: ADD YOUR CUSTOM METRICS BELOW
    # ============================================================
    # Example metrics you might want to add:
    #
    # 1. Whitening layer analysis:
    #    W = model.whiten.weight
    #    logs["whiten_weight_norm"] = float(W.norm().item())
    #
    # 2. Covariance condition number (if you compute patch covariance):
    #    logs["condition_number"] = float(cond_num)
    #    logs["off_diagonal_energy"] = float(off_diag)
    #    logs["isotropy_error"] = float(iso_err)
    #
    # 3. Orthogonality metrics:
    #    logs["orthogonality_error"] = compute_orth_error(W)
    #
    # ============================================================

    return logs


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


def infer(model, loader, tta_level=0, tta_uncertain_quantile=0.25):
    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    @CustomTorchCompile(fullgraph=True)
    def _get_tta_logits(model, images_batch, pad):
        batch_size = images_batch.shape[0]
        padded_inputs = F.pad(images_batch, (pad,) * 4, "reflect")
        crop_tl = padded_inputs[:, :, 0:32, 0:32]
        crop_br = padded_inputs[:, :, 2:34, 2:34]
        base_views = torch.cat([images_batch, crop_tl, crop_br], dim=0)
        flipped_views = base_views.flip(-1)
        combined_inputs = torch.cat([base_views, flipped_views], dim=0)
        combined_logits = model(combined_inputs)
        num_views = combined_inputs.shape[0] // batch_size
        reshaped_logits = combined_logits.view(num_views, batch_size, -1)
        averaged_logits = reshaped_logits.mean(dim=0)
        return averaged_logits

    @CustomTorchCompile()
    def tta(model, test_images) -> torch.Tensor:
        with torch.no_grad():
            model.eval()
            # device = test_images.device
            B = 2000
            pad = 1
            n = test_images.shape[0]
            all_logits_list = []
            for inputs_batch in test_images.split(B):
                inputs_batch = inputs_batch.contiguous(
                    memory_format=torch.channels_last
                )
                all_logits_list.append(model(inputs_batch).clone())
            initial_logits = torch.cat(all_logits_list, dim=0)
            probs = F.softmax(initial_logits, dim=1)
            confidences, _ = probs.max(dim=1)
            k_uncertain = int(n * tta_uncertain_quantile)
            if k_uncertain == 0:
                return initial_logits
            _, uncertain_indices = torch.topk(
                confidences, k_uncertain, largest=False, sorted=False
            )

            tta_logits_parts = []
            tta_batch_size = 2000
            for i in range(0, k_uncertain, tta_batch_size):
                cur_batch_size = min(tta_batch_size, k_uncertain - i)
                batch_indices = uncertain_indices[i : i + cur_batch_size]
                images_batch = test_images[batch_indices]
                logits_batch = _get_tta_logits(
                    model,
                    images_batch.contiguous(memory_format=torch.channels_last),
                    pad,
                )
                tta_logits_parts.append(logits_batch)

            if tta_logits_parts:
                all_tta_logits_for_uncertain = torch.cat(tta_logits_parts, dim=0)
                final_logits = initial_logits.clone()
                final_logits[uncertain_indices] = all_tta_logits_for_uncertain
                return final_logits
            return initial_logits

    test_images = loader.normalize(loader.images)
    if tta_level < 2:
        model.eval()
        if tta_level == 0:
            infer_fn = infer_basic
        elif tta_level == 1:
            infer_fn = infer_mirror
        else:
            infer_fn = None
        with torch.no_grad():
            return torch.cat(
                [infer_fn(inputs, model) for inputs in test_images.split(2000)]
            )
    else:  # tta_level == 2
        return tta(model, test_images)


def evaluate(model, loader, tta_level=0, tta_uncertain_quantile=0.25):
    logits = infer(model, loader, tta_level, tta_uncertain_quantile)
    return (logits.argmax(1) == loader.labels).float().mean().item()


############################################
#                Training                  #
############################################


def main(run, model, config: SpeedrunConfig):
    """
    Main training function.

    Args:
        run: Run identifier ("warmup" or run number for timed run)
        model: The model to train
    """
    is_warmup = run == "warmup"
    training_batch_size = config.training_batch_size
    bias_lr = config.bias_lr
    head_lr = config.head_lr
    wd = config.weight_decay_scale * training_batch_size
    test_loader = CifarLoader(config.data_dir, train=False, batch_size=2000)
    train_loader = CifarLoader(
        config.data_dir,
        train=True,
        batch_size=training_batch_size,
        aug={
            "flip": True,
            "translate": config.translate,
            "color_jitter": {
                "enabled": True,
                "brightness_range": config.brightness_range,
                "contrast_range": config.contrast_range,
            },
        },
    )
    if is_warmup:
        train_loader.labels = torch.randint(
            0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device
        )
        train_loader.images = torch.randn_like(
            train_loader.images, device=train_loader.images.device
        )
        test_loader.labels = torch.randint(
            0, 10, size=(len(test_loader.labels),), device=test_loader.labels.device
        )
        test_loader.images = torch.randn_like(
            test_loader.images, device=test_loader.images.device
        )
    total_train_steps = ceil(config.train_epochs * len(train_loader))
    whiten_bias_train_steps = ceil(config.whiten_bias_epochs * len(train_loader))

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
        param_configs, momentum=config.sgd_momentum, nesterov=True, fused=True
    )
    optimizer2 = Muon(
        filter_params,
        lr=config.muon_lr,
        momentum=config.muon_momentum,
        nesterov=True,
        norm_freq=4,
        total_train_steps=total_train_steps,
        weight_decay=wd,
    )
    optimizer2.param_groups[0]["momentum_buffer_dtype"] = torch.half
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

    # Collect simple per-epoch logs for downstream analysis
    all_logs = []

    # Initialize the whitening layer using training images
    start_timer()
    with torch.no_grad():
        train_images = train_loader.normalize(train_loader.images[:960])
        model.init_whiten(train_images)
    stop_timer()

    # Precompute LR factors to reduce computation in training loop
    lr_factor1_base = 1.0 / max(1, whiten_bias_train_steps)
    lr_factor2_base = 1.0 / total_train_steps

    # Precompute some values to reduce computation in training loop
    lr_factor1_initial = optimizer1.param_groups[0]["initial_lr"]
    lr_factors2_initial = [
        group["initial_lr"]
        for group in optimizer1.param_groups[1:] + optimizer2.param_groups
    ]

    # Compile the forward pass function with reduced overhead
    @CustomTorchCompile(mode="max-autotune", fullgraph=True)
    def forward_step(inputs, labels, whiten_bias_grad):
        outputs = model(inputs, whiten_bias_grad=whiten_bias_grad)
        loss = F.cross_entropy(outputs, labels, label_smoothing=config.label_smoothing, reduction="sum")
        return loss, outputs

    for epoch in range(ceil(total_train_steps / len(train_loader))):
        ####################
        #     Training     #
        ####################
        start_timer()
        model.train()

        for inputs, labels in train_loader:
            # Determine if we should train whiten bias
            whiten_bias_grad = step < whiten_bias_train_steps

            # Execute training step
            loss, outputs = forward_step(inputs, labels, whiten_bias_grad)
            loss.backward()

            # Update learning rates more efficiently
            lr_factor1 = 1 - step * lr_factor1_base
            lr_factor2 = 1 - step * lr_factor2_base

            # Apply learning rates in a fused way
            optimizer1.param_groups[0]["lr"] = lr_factor1_initial * lr_factor1
            for i, group in enumerate(
                optimizer1.param_groups[1:] + optimizer2.param_groups
            ):
                group["lr"] = lr_factors2_initial[i] * lr_factor2

            # Optimizer steps
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0, tta_uncertain_quantile=TTA_UNCERTAIN_QUANTILE)
        print_training_details(locals(), is_final_entry=False)
        run = None  # Only print the run number once

        # Only record logs for non-warmup runs (warmup is just for compilation)
        if not is_warmup:
            # Use compute_logs to aggregate metrics - this is the extension point
            epoch_log = compute_logs(
                model=model,
                val_acc=val_acc,
                train_acc=train_acc,
                time_seconds=time_seconds,
            )
            all_logs.append(epoch_log)

        if step >= total_train_steps:
            break

    if is_warmup:
        return 0.0, []

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2, tta_uncertain_quantile=TTA_UNCERTAIN_QUANTILE)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    # Attach TTA accuracy to the last epoch log if available
    if all_logs:
        all_logs[-1]["tta_val_acc"] = float(tta_val_acc)

    return tta_val_acc, all_logs


def parse_args() -> SpeedrunConfig:
    parser = argparse.ArgumentParser(
        description="Autotune-native CIFAR-10 speedrun based on the Agentic Scientist baseline."
    )
    parser.add_argument("--training-batch-size", type=bounded_int(64, 4096), default=1536)
    parser.add_argument("--bias-lr", type=bounded_float(0.001, 0.2), default=0.0573)
    parser.add_argument("--head-lr", type=bounded_float(0.01, 1.0), default=0.5415)
    parser.add_argument("--weight-decay-scale", type=bounded_float(1e-08, 1e-04), default=1.0418e-06)
    parser.add_argument("--sgd-momentum", type=unit_interval, default=0.825)
    parser.add_argument("--muon-lr", type=bounded_float(0.01, 0.5), default=0.205)
    parser.add_argument("--muon-momentum", type=unit_interval, default=0.655)
    parser.add_argument("--train-epochs", type=bounded_float(1.0, 12.0), default=7.65)
    parser.add_argument("--whiten-bias-epochs", type=bounded_float(0.0, 2.0), default=0.2)
    parser.add_argument("--label-smoothing", type=unit_interval, default=0.09)
    parser.add_argument("--translate", type=bounded_int(0, 4), default=2)
    parser.add_argument("--brightness-range", type=bounded_float(0.0, 0.5), default=0.1399)
    parser.add_argument("--contrast-range", type=bounded_float(0.0, 0.5), default=0.1308)
    args = parser.parse_args()
    return SpeedrunConfig(
        training_batch_size=args.training_batch_size,
        bias_lr=args.bias_lr,
        head_lr=args.head_lr,
        weight_decay_scale=args.weight_decay_scale,
        sgd_momentum=args.sgd_momentum,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        train_epochs=args.train_epochs,
        whiten_bias_epochs=args.whiten_bias_epochs,
        label_smoothing=args.label_smoothing,
        translate=args.translate,
        brightness_range=args.brightness_range,
        contrast_range=args.contrast_range,
        data_dir=os.environ.get("CIFAR10_SPEEDRUN_DATA_DIR", DEFAULT_DATA_DIR),
    )


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError(f"expected value in [0, 1], got {value}")
    return parsed


def bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"expected integer in [{minimum}, {maximum}], got {value}")
        return parsed

    return parse


def bounded_float(minimum: float, maximum: float) -> Callable[[str], float]:
    def parse(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"expected finite float in [{minimum}, {maximum}], got {value}")
        return parsed

    return parse


def read_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return bounded_int(1, 1000)(raw)


def read_positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return bounded_float(0.1, 3600.0)(raw)


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The CIFAR-10 speedrun example requires a CUDA-capable GPU")


def config_fingerprint(config: SpeedrunConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_results_json(path: str, payload: dict[str, object]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def optional_results_paths(config: SpeedrunConfig) -> list[str]:
    paths = []
    results_path = os.environ.get("CIFAR10_SPEEDRUN_RESULTS_JSON")
    if results_path:
        paths.append(results_path)
    results_dir = os.environ.get("CIFAR10_SPEEDRUN_RESULTS_DIR")
    if results_dir:
        paths.append(
            os.path.join(
                results_dir,
                f"{config_fingerprint(config)}-{os.getpid()}.json",
            )
        )
    return paths


def write_optional_results_json(results: dict[str, object], config: SpeedrunConfig) -> None:
    paths = optional_results_paths(config)
    if not paths:
        return
    payload = {
        **results,
        "config": asdict(config),
        "config_fingerprint": config_fingerprint(config),
    }
    for path in paths:
        write_results_json(path, payload)


def compute_score(acc: float, mean_time: float) -> float:
    accuracy_gap = max(0.0, ACCURACY_THRESHOLD - acc)
    penalty = (1e6 ** (20 * accuracy_gap)) - 1
    return -mean_time - penalty


def run_experiment(config: SpeedrunConfig, seed=42, num_runs=DEFAULT_NUM_RUNS, max_time_per_run=DEFAULT_MAX_TIME_PER_RUN):
    """
    Run the experiment with three phases:
    1. Warmup: JIT compilation (timing not counted)
    2. Timed runs: Multiple clean runs for accurate speedrun timing

    Args:
        seed: Random seed (unused currently, for interface compatibility)
        num_runs: Number of timed runs for statistics (default: 100)
        max_time_per_run: Maximum allowed time per run in seconds (default: 3.0).
                          If a run exceeds this, we terminate early as too slow.
    """
    require_cuda()
    torch.manual_seed(seed)
    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)

    # Run one uncompiled forward pass before compilation.
    # This ensures coverage tools (e.g., `coverage` library) can track the forward method's
    # source lines, since they only observe interpreted (non-compiled) execution.
    with torch.no_grad():
        dummy_input = torch.zeros(1, 3, 32, 32, device="cuda", dtype=torch.float16).to(
            memory_format=torch.channels_last
        )
        model(dummy_input)

    model.compile(mode="max-autotune")

    print_columns(logging_columns_list, is_head=True)

    # Phase 1: Warmup run (for JIT compilation)
    main("warmup", model, config)

    # Phase 2: Timed runs - multiple clean runs for official speedrun scoring
    print(f"\n--- Timed Runs ({num_runs} runs for speedrun scoring) ---")
    all_tta_val_accs = []
    all_times = []
    all_timed_logs = []
    terminated_early = False

    for run in range(num_runs):
        # Clear GPU cache and synchronize between runs (matching official speedrun)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        tta_val_acc, timed_logs = main(run + 1, model, config)
        all_tta_val_accs.append(tta_val_acc)

        if timed_logs:
            run_time = timed_logs[-1]["time_seconds"]
            all_times.append(run_time)
            all_timed_logs.append(timed_logs)

            # Early termination if run is too slow
            if run_time > max_time_per_run:
                print(
                    f"\n*** EARLY TERMINATION: Run {run + 1} took {run_time:.2f}s "
                    f"(> {max_time_per_run:.1f}s threshold) - too slow for speedrun ***"
                )
                terminated_early = True
                break

    # Compute statistics across completed runs
    actual_runs = len(all_tta_val_accs)
    mean_acc = np.mean(all_tta_val_accs)
    std_acc = np.std(all_tta_val_accs) if actual_runs > 1 else float("nan")
    mean_time = np.mean(all_times) if all_times else float("nan")
    std_time = np.std(all_times) if len(all_times) > 1 else float("nan")

    if actual_runs > 1:
        t_crit = scipy.stats.t.ppf(0.975, df=actual_runs - 1)
        ci95_acc = t_crit * std_acc / np.sqrt(actual_runs)
        ci95_time = t_crit * std_time / np.sqrt(actual_runs)
    else:
        ci95_acc = float("nan")
        ci95_time = float("nan")

    print(f"\n{'='*60}")
    if terminated_early:
        print(
            f"Results over {actual_runs}/{num_runs} runs (TERMINATED EARLY - TOO SLOW):"
        )
    else:
        print(f"Results over {actual_runs} runs:")
    print(f"  TTA Accuracy: Mean={mean_acc:.4f}  Std={std_acc:.4f}  95% CI=[{mean_acc - ci95_acc:.4f}, {mean_acc + ci95_acc:.4f}]")
    print(f"  Time (sec):   Mean={mean_time:.4f}  Std={std_time:.4f}  95% CI=[{mean_time - ci95_time:.4f}, {mean_time + ci95_time:.4f}]")
    print(f"{'='*60}")

    results = {
        "combined_score": compute_score(float(mean_acc), float(mean_time)),
        "tta_val_acc_mean": float(mean_acc),
        "tta_val_acc_std": float(std_acc) if not np.isnan(std_acc) else None,
        "tta_val_acc_ci95_lower": float(mean_acc - ci95_acc) if not np.isnan(ci95_acc) else None,
        "tta_val_acc_ci95_upper": float(mean_acc + ci95_acc) if not np.isnan(ci95_acc) else None,
        "time_mean": float(mean_time) if not np.isnan(mean_time) else None,
        "time_std": float(std_time) if not np.isnan(std_time) else None,
        "time_ci95_lower": float(mean_time - ci95_time) if not np.isnan(ci95_time) else None,
        "time_ci95_upper": float(mean_time + ci95_time) if not np.isnan(ci95_time) else None,
        "num_completed_runs": actual_runs,
        "terminated_early": terminated_early,
    }
    write_optional_results_json(results, config)
    print(f"autotune_metric={results['combined_score']}", flush=True)

    return results


if __name__ == "__main__":
    run_experiment(
        parse_args(),
        num_runs=read_positive_int_env("CIFAR10_SPEEDRUN_NUM_RUNS", DEFAULT_NUM_RUNS),
        max_time_per_run=read_positive_float_env("CIFAR10_SPEEDRUN_MAX_TIME_PER_RUN", DEFAULT_MAX_TIME_PER_RUN),
    )
