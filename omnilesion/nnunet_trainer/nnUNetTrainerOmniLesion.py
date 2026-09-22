"""OmniLesion trainer for nnU-Net v2 (2.6.0).

One module, self-contained, to be copied into ``nnunetv2/training/nnUNetTrainer/`` of an nnU-Net v2 installation
(``python -m omnilesion.nnunet_trainer.install``). It defines everything the submitted model needs on top of the stock
trainer:

* a three-channel input built on the fly from the normalised CT (native window, soft tissue -150..250 HU, lung
  -1000..400 HU), see :class:`MultiWindowCT`;
* the component-balanced Dice term added at native resolution to nnU-Net's deep-supervised Dice + cross-entropy, see
  :class:`ComponentBalancedDice` (CUDA, CuPy connected components) and its CPU twin;
* per-case sampling weights read from a JSON table (organ-family-balanced sampling), see
  :meth:`nnUNetTrainerOmniLesion.get_dataloaders`;
* a training split read from a JSON file with ``train`` and ``sentinel`` case lists instead of nnU-Net's fold split;
* SGD with Nesterov momentum 0.99 and a polynomial learning-rate decay with configurable exponent, and a configurable
  number of iterations per epoch.

Every knob is an environment variable with the value used for the submission as default:

    OMNILESION_SPLIT_JSON          path of the split file (required; keys ``train`` and ``sentinel``)
    OMNILESION_CASE_WEIGHTS_JSON   path of the per-case weight table (required; key ``weights``)
    OMNILESION_EPOCHS              4000
    OMNILESION_ITERS_PER_EPOCH     750   (nnU-Net default 250; validation iterations scale with it)
    OMNILESION_LR                  0.01
    OMNILESION_POLY_EXPONENT       2.0   (nnU-Net default 0.9)
    OMNILESION_COMPONENT_WEIGHT    1.0   weight of the component-balanced term
    OMNILESION_MAX_COMPONENTS      48    components per patch entering the term (largest kept)
    OMNILESION_SEED                unset; set an integer to seed Python, NumPy and torch before construction
    OMNILESION_FOREGROUND_MEAN_HU  -44.66527557373047  foreground CT statistics of the corpus (see MultiWindowCT)
    OMNILESION_FOREGROUND_STD_HU   245.2626190185547
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from scipy import ndimage
from torch import nn

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cupy_ndimage
except ImportError:  # CPU-only hosts can still import the module (and run the CPU loss in tests).
    cp = None
    cupy_ndimage = None


# ---------------------------------------------------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------------------------------------------------

class ComponentBalancedDiceCPU(nn.Module):
    """Dice averaged over the connected components of the reference, each evaluated in its own padded bounding box.

    Standard Dice is volume-weighted, so one large lesion dominates a patch with many small ones. This term gives every
    reference component equal weight and evaluates it in a local context (bounding box padded by ``context_margin``
    voxels), so false positives next to a component are penalised as well. Components are 26-connected; when a patch
    holds more than ``max_components`` the largest are kept. Reference implementation on the CPU (scipy); the CUDA
    version below is numerically equivalent.
    """

    def __init__(self, context_margin: int = 8, max_components: int = 48, smooth: float = 1e-5):
        super().__init__()
        self.context_margin = int(context_margin)
        self.max_components = int(max_components)
        self.smooth = float(smooth)
        self.structure = np.ones((3, 3, 3), dtype=np.uint8)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or target.ndim != 5:
            raise ValueError("ComponentBalancedDice expects 3D logits [B, C, D, H, W] and targets [B, 1, D, H, W]")
        probability = torch.softmax(logits, dim=1)[:, 1]
        binary = (target[:, 0] > 0).detach().to(device="cpu", dtype=torch.uint8).numpy()
        losses = []
        spatial_shape = binary.shape[1:]
        for batch_index, mask in enumerate(binary):
            labels, n_components = ndimage.label(mask, structure=self.structure)
            if n_components == 0:
                continue
            component_ids = np.arange(1, n_components + 1)
            sizes = np.bincount(labels.ravel(), minlength=n_components + 1)[1:]
            if n_components > self.max_components:
                component_ids = component_ids[np.argsort(sizes)[-self.max_components:]]
            for component_id in component_ids:
                component = labels == component_id
                coordinates = np.where(component)
                starts = [max(0, int(axis.min()) - self.context_margin) for axis in coordinates]
                stops = [min(spatial_shape[axis], int(coordinates[axis].max()) + 1 + self.context_margin)
                         for axis in range(3)]
                region = tuple(slice(starts[axis], stops[axis]) for axis in range(3))
                gt = torch.as_tensor(component[region], device=logits.device, dtype=probability.dtype)
                pred = probability[(batch_index, *region)]
                intersection = (pred * gt).sum()
                denominator = pred.sum() + gt.sum()
                losses.append(1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth))
        return torch.stack(losses).mean() if losses else logits.new_zeros(())


class ComponentBalancedDice(nn.Module):
    """CUDA version of :class:`ComponentBalancedDiceCPU`.

    Connected components are labelled on the GPU with CuPy (26-connectivity); component selection, bounding boxes and
    the Dice reductions stay in PyTorch on the same device, so nothing is copied to the host during training. The padded
    bounding box is applied as a boolean context mask, which is mathematically identical to slicing.
    """

    def __init__(self, context_margin: int = 8, max_components: int = 48, smooth: float = 1e-5):
        super().__init__()
        self.context_margin = int(context_margin)
        self.max_components = int(max_components)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 5 or target.ndim != 5:
            raise ValueError("ComponentBalancedDice expects 3D logits [B, C, D, H, W] and targets [B, 1, D, H, W]")
        if not logits.is_cuda:
            raise RuntimeError("ComponentBalancedDice runs on CUDA only; use ComponentBalancedDiceCPU on the CPU")
        if cp is None or cupy_ndimage is None:
            raise RuntimeError("ComponentBalancedDice needs CuPy (cupy-cuda12x) in the training environment")

        probability = torch.softmax(logits, dim=1)[:, 1]
        binary = (target[:, 0] > 0).to(dtype=torch.uint8)
        _, depth, height, width = binary.shape
        device = logits.device
        axis_z = torch.arange(depth, device=device, dtype=torch.long)
        axis_y = torch.arange(height, device=device, dtype=torch.long)
        axis_x = torch.arange(width, device=device, dtype=torch.long)
        all_component_losses = []
        all_valid_components = []

        for batch_index in range(binary.shape[0]):
            labels_cupy, number_of_components = cupy_ndimage.label(
                cp.from_dlpack(binary[batch_index].detach()),
                structure=cp.ones((3, 3, 3), dtype=cp.uint8),
            )
            retained = min(int(number_of_components), self.max_components)
            if retained == 0:
                continue
            labels = torch.from_dlpack(labels_cupy).to(dtype=torch.long)
            sizes = torch.bincount(labels.reshape(-1))
            component_sizes = sizes[1:]
            selected_sizes, selected_indices = torch.topk(component_sizes, k=retained)
            selected_ids = selected_indices + 1

            component = labels.unsqueeze(0) == selected_ids[:, None, None, None]
            valid = selected_sizes > 0
            has_z = component.any(dim=(2, 3))
            has_y = component.any(dim=(1, 3))
            has_x = component.any(dim=(1, 2))
            min_z = torch.where(has_z, axis_z[None], depth).amin(dim=1)
            max_z = torch.where(has_z, axis_z[None], -1).amax(dim=1)
            min_y = torch.where(has_y, axis_y[None], height).amin(dim=1)
            max_y = torch.where(has_y, axis_y[None], -1).amax(dim=1)
            min_x = torch.where(has_x, axis_x[None], width).amin(dim=1)
            max_x = torch.where(has_x, axis_x[None], -1).amax(dim=1)
            start_z = (min_z - self.context_margin).clamp(0, depth)
            stop_z = (max_z + 1 + self.context_margin).clamp(0, depth)
            start_y = (min_y - self.context_margin).clamp(0, height)
            stop_y = (max_y + 1 + self.context_margin).clamp(0, height)
            start_x = (min_x - self.context_margin).clamp(0, width)
            stop_x = (max_x + 1 + self.context_margin).clamp(0, width)
            context = (
                (axis_z[None, :, None, None] >= start_z[:, None, None, None])
                & (axis_z[None, :, None, None] < stop_z[:, None, None, None])
                & (axis_y[None, None, :, None] >= start_y[:, None, None, None])
                & (axis_y[None, None, :, None] < stop_y[:, None, None, None])
                & (axis_x[None, None, None, :] >= start_x[:, None, None, None])
                & (axis_x[None, None, None, :] < stop_x[:, None, None, None])
            )
            gt = component.to(dtype=probability.dtype)
            pred = probability[batch_index].unsqueeze(0) * context.to(dtype=probability.dtype)
            intersection = (pred * gt).sum(dim=(1, 2, 3))
            denominator = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
            all_component_losses.append(1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth))
            all_valid_components.append(valid.to(dtype=probability.dtype))

        if not all_component_losses:
            return logits.new_zeros(())
        loss_values = torch.cat(all_component_losses)
        valid_values = torch.cat(all_valid_components)
        return (loss_values * valid_values).sum() / valid_values.sum().clamp_min(1.0)


class DiceCEPlusComponentDice(nn.Module):
    """nnU-Net's deep-supervised Dice + cross-entropy plus the component-balanced term at native resolution only."""

    def __init__(self, baseline_loss: nn.Module, component_loss: nn.Module, component_weight: float):
        super().__init__()
        self.baseline_loss = baseline_loss
        self.component = component_loss
        self.component_weight = float(component_weight)

    def forward(self, prediction, target) -> torch.Tensor:
        standard_loss = self.baseline_loss(prediction, target)
        native_prediction = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        native_target = target[0] if isinstance(target, (tuple, list)) else target
        return standard_loss + self.component_weight * self.component(native_prediction, native_target)


# ---------------------------------------------------------------------------------------------------------------------
# Network input
# ---------------------------------------------------------------------------------------------------------------------

class MultiWindowCT(nn.Module):
    """Expand the normalised CT channel into three channels: native, soft-tissue window and lung window.

    nnU-Net's CT normalisation feeds ``(HU - mean) / std`` with the foreground statistics of the training corpus. The
    wrapper undoes it to recover HU, clips two fixed windows and maps each to [-1, 1], then concatenates all three and
    calls the backbone. The statistics are stored as buffers, so a checkpoint carries them.
    """

    def __init__(self, backbone: nn.Module, mean_hu: float, std_hu: float):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean_hu", torch.tensor(float(mean_hu)), persistent=True)
        self.register_buffer("std_hu", torch.tensor(float(std_hu)), persistent=True)

    @property
    def decoder(self):
        return self.backbone.decoder

    @staticmethod
    def window(hu: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        center = (lower + upper) / 2.0
        half_width = (upper - lower) / 2.0
        return ((hu.clamp(lower, upper) - center) / half_width).to(dtype=hu.dtype)

    def forward(self, normalized_ct: torch.Tensor):
        hu = normalized_ct.float() * self.std_hu + self.mean_hu
        soft_tissue = self.window(hu, -150.0, 250.0)
        lung = self.window(hu, -1000.0, 400.0)
        return self.backbone(torch.cat((normalized_ct, soft_tissue, lung), dim=1))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# Foreground intensity statistics of the training corpus (``foreground_intensity_properties_per_channel`` in the plan
# generated by scripts/02_plan_and_preprocess.sh). ``build_network_architecture`` is a static method without access to
# the plan, so the values are fixed here; override them with OMNILESION_FOREGROUND_MEAN_HU / _STD_HU when training on a different corpus
# (scripts/make_resenc_plans.py prints the values of a new plan).
FOREGROUND_MEAN_HU = _env_float("OMNILESION_FOREGROUND_MEAN_HU", -44.66527557373047)
FOREGROUND_STD_HU = _env_float("OMNILESION_FOREGROUND_STD_HU", 245.2626190185547)


# ---------------------------------------------------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------------------------------------------------

class nnUNetTrainerOmniLesion(nnUNetTrainer):
    """The OmniLesion training recipe on top of the stock nnU-Net v2 trainer (see the module docstring)."""

    component_weight: float = _env_float("OMNILESION_COMPONENT_WEIGHT", 1.0)
    max_components: int = _env_int("OMNILESION_MAX_COMPONENTS", 48)

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        seed_value = os.environ.get("OMNILESION_SEED")
        if seed_value is not None:
            seed = int(seed_value)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = _env_int("OMNILESION_EPOCHS", 4000)
        self.schedule_epochs = self.num_epochs
        self.initial_lr = _env_float("OMNILESION_LR", 0.01)
        self.poly_exponent = _env_float("OMNILESION_POLY_EXPONENT", 2.0)
        # An nnU-Net epoch is a fixed number of iterations, not a pass over the data, and the learning rate steps once
        # per epoch, so total updates = epochs x iterations can be set independently of the schedule length. Validation
        # iterations scale with the same factor.
        iterations = _env_int("OMNILESION_ITERS_PER_EPOCH", 750)
        if iterations != 250:
            self.num_iterations_per_epoch = iterations
            self.num_val_iterations_per_epoch = max(1, int(round(50 * iterations / 250)))
        self.split_json = os.environ.get("OMNILESION_SPLIT_JSON")
        self.case_weights_json = os.environ.get("OMNILESION_CASE_WEIGHTS_JSON")

    # ----- optimiser and schedule --------------------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.schedule_epochs, exponent=self.poly_exponent)
        self.print_to_log_file(f"PolyLR: initial_lr={self.initial_lr} horizon={self.schedule_epochs} "
                               f"exponent={self.poly_exponent}; {self.num_iterations_per_epoch} iterations per epoch")
        return optimizer, scheduler

    # ----- split ---------------------------------------------------------------------------------------------------
    def do_split(self):
        if not self.split_json:
            raise RuntimeError("set OMNILESION_SPLIT_JSON to a JSON file with 'train' and 'sentinel' case lists")
        split = json.loads(Path(self.split_json).read_text())
        training = list(split["train"])
        validation = list(split["sentinel"])
        overlap = set(training).intersection(validation)
        if overlap:
            raise RuntimeError(f"split leakage: {len(overlap)} cases are in both train and sentinel")
        self.print_to_log_file(f"Split {self.split_json}: train={len(training)}, sentinel={len(validation)}; "
                               f"stop epoch={self.num_epochs}, schedule horizon={self.schedule_epochs}.")
        return training, validation

    # ----- network -------------------------------------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        backbone = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels + 2, num_output_channels, enable_deep_supervision)
        return MultiWindowCT(backbone, mean_hu=FOREGROUND_MEAN_HU, std_hu=FOREGROUND_STD_HU)

    # ----- loss ----------------------------------------------------------------------------------------------------
    def _build_loss(self):
        baseline = super()._build_loss()  # deep-supervised Dice + CE with nnU-Net's default scale weights
        component = ComponentBalancedDice(context_margin=8, max_components=self.max_components)
        return DiceCEPlusComponentDice(baseline, component, self.component_weight)

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(f"Loss: Dice + CE (deep supervision) + {self.component_weight} x component-balanced "
                               f"Dice at native resolution (context margin 8, up to {self.max_components} components).")

    # ----- weighted case sampling ----------------------------------------------------------------------------------
    def _sampling_probabilities(self, identifiers) -> np.ndarray:
        if not self.case_weights_json:
            raise RuntimeError("set OMNILESION_CASE_WEIGHTS_JSON to the per-case weight table "
                               "(scripts/06_build_case_weights.py)")
        payload = json.loads(Path(self.case_weights_json).read_text())
        table = payload["weights"]
        identifiers = list(identifiers)
        present = set(identifiers)
        orphans = [case for case in table if case not in present]
        if orphans:
            raise RuntimeError(f"{len(orphans)} weighted cases are absent from the training split, e.g. {orphans[:5]}; "
                               f"the weight table must be built from the same split")
        weights = np.array([float(table.get(identifier, 1.0)) for identifier in identifiers])
        boosted = weights > 1.0
        if not boosted.any():
            raise RuntimeError(f"{self.case_weights_json} boosts no case in this split")
        probabilities = weights / weights.sum()
        cohorts = Counter(identifier.rsplit("_", 1)[0] for identifier, flag in zip(identifiers, boosted) if flag)
        self.print_to_log_file(f"Case weights from {self.case_weights_json}: {int(boosted.sum())} of "
                               f"{len(identifiers)} cases above 1.0 (max {weights.max():.2f}); "
                               f"cohorts {cohorts.most_common(8)}")
        return probabilities

    def get_dataloaders(self):
        """nnU-Net's loader construction with per-case sampling probabilities on the training loader only."""
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation, dummy_2d, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size())
        regions = self.label_manager.foreground_regions if self.label_manager.has_regions else None
        train_transforms = self.get_training_transforms(
            patch_size, rotation, deep_supervision_scales, mirror_axes, dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels, regions=regions,
            ignore_label=self.label_manager.ignore_label)
        validation_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels, regions=regions,
            ignore_label=self.label_manager.ignore_label)
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        probabilities = self._sampling_probabilities(dataset_tr.identifiers)

        train_loader = nnUNetDataLoader(
            dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=probabilities, pad_sides=None, transforms=train_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)
        validation_loader = nnUNetDataLoader(
            dataset_val, self.batch_size, self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None, transforms=validation_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)
        workers = get_allowed_n_proc_DA()
        if workers == 0:
            return SingleThreadedAugmenter(train_loader, None), SingleThreadedAugmenter(validation_loader, None)
        train = NonDetMultiThreadedAugmenter(data_loader=train_loader, transform=None, num_processes=workers,
                                             num_cached=max(6, workers // 2), seeds=None,
                                             pin_memory=self.device.type == "cuda", wait_time=0.002)
        validation = NonDetMultiThreadedAugmenter(data_loader=validation_loader, transform=None,
                                                  num_processes=max(1, workers // 2),
                                                  num_cached=max(3, workers // 4), seeds=None,
                                                  pin_memory=self.device.type == "cuda", wait_time=0.002)
        _ = next(train)
        _ = next(validation)
        return train, validation
