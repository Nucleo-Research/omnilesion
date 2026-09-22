"""Small 3D U-Net that segments one lung nodule in a 65^3 patch sampled at 1 mm isotropic spacing around a proposal.

Inference uses only the CT, the proposal centre and the image spacing. The mask is thresholded and reduced to the
26-connected component containing the patch centre (or the nearest one), then projected back onto the native grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

DEFAULT_PATCH_SIZE = 65
DEFAULT_FILL_HU = -1000.0


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = max(group for group in range(1, min(8, out_channels) + 1) if out_channels % group == 0)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    """Three encoder levels plus bottleneck (base, 2x, 4x, 8x channels), max-pool down, trilinear up, 1x1x1 head."""

    def __init__(self, in_channels: int = 1, base_channels: int = 12):
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bottleneck = ConvBlock(b * 4, b * 8)
        self.dec3 = ConvBlock(b * 8 + b * 4, b * 4)
        self.dec2 = ConvBlock(b * 4 + b * 2, b * 2)
        self.dec1 = ConvBlock(b * 2 + b, b)
        self.out = nn.Conv3d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        b = self.bottleneck(F.max_pool3d(e3, 2))
        d3 = F.interpolate(b, size=e3.shape[2:], mode="trilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def normalize_hu(volume: np.ndarray) -> np.ndarray:
    """Clip to [-1000, 600] HU and map linearly to [-1, 1]."""
    clipped = np.clip(volume, -1000.0, 600.0)
    return ((clipped + 1000.0) / 1600.0 * 2.0 - 1.0).astype(np.float32)


def center_component(mask: np.ndarray) -> np.ndarray:
    """Keep the 26-connected component containing the patch centre, else the one nearest to it."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    center = tuple(int((dim - 1) / 2) for dim in mask.shape)
    center_label = int(labels[center])
    if center_label > 0:
        return labels == center_label
    coords = np.argwhere(labels > 0)
    if coords.size == 0:
        return mask
    center_arr = np.array(center, dtype=float)
    best_label, best_distance = 0, float("inf")
    for label in range(1, count + 1):
        label_coords = coords[labels[tuple(coords.T)] == label]
        if label_coords.size == 0:
            continue
        distance = float(np.min(np.linalg.norm(label_coords - center_arr, axis=1)))
        if distance < best_distance:
            best_distance, best_label = distance, label
    return labels == best_label if best_label > 0 else mask


def extract_isotropic_patch(volume_xyz: np.ndarray, center_xyz: Tuple[float, float, float],
                            spacing_xyz: Tuple[float, float, float], patch_size: int = DEFAULT_PATCH_SIZE,
                            output_spacing_mm: float = 1.0, fill_hu: float = DEFAULT_FILL_HU) -> np.ndarray:
    """Sample a centred isotropic patch (linear interpolation) from a native CT volume indexed [x, y, z]."""
    if volume_xyz.ndim != 3:
        raise ValueError("volume_xyz must be 3D")
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd so the centre is a voxel")
    offsets_mm = (np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0) * output_spacing_mm
    xs = center_xyz[0] + offsets_mm / spacing_xyz[0]
    ys = center_xyz[1] + offsets_mm / spacing_xyz[1]
    zs = center_xyz[2] + offsets_mm / spacing_xyz[2]
    grid = np.meshgrid(xs, ys, zs, indexing="ij")
    patch = ndimage.map_coordinates(volume_xyz.astype(np.float32), grid, order=1, mode="constant", cval=float(fill_hu))
    return patch.astype(np.float32)


def patch_mask_to_native(patch_mask: np.ndarray, volume_shape: Tuple[int, int, int],
                         center_xyz: Tuple[float, float, float], spacing_xyz: Tuple[float, float, float],
                         output_spacing_mm: float = 1.0) -> np.ndarray:
    """Project an isotropic patch mask back onto the nearest native voxels."""
    full_mask = np.zeros(volume_shape, dtype=bool)
    coords = np.argwhere(patch_mask.astype(bool))
    if coords.size == 0:
        return full_mask
    patch_center = (np.array(patch_mask.shape, dtype=np.float32) - 1.0) / 2.0
    offsets_mm = (coords.astype(np.float32) - patch_center) * output_spacing_mm
    native = np.rint(np.array(center_xyz, dtype=np.float32)
                     + offsets_mm / np.array(spacing_xyz, dtype=np.float32)).astype(int)
    valid = np.all((native >= 0) & (native < np.array(volume_shape)), axis=1)
    native = native[valid]
    if native.size:
        full_mask[tuple(native.T)] = True
    return full_mask


class NoduleUNetSegmenter:
    def __init__(self, model: nn.Module, threshold: float, device: Optional[torch.device] = None,
                 keep_center_component: bool = True):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.threshold = float(threshold)
        self.keep_center_component = keep_center_component

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, manifest_path: str | Path,
                        device: Optional[torch.device] = None) -> "NoduleUNetSegmenter":
        """``manifest_path`` is the JSON written by scripts/lung_cascade/train_segmentor.py."""
        metadata = json.load(open(manifest_path))
        model_meta = metadata.get("model", {})
        model = UNet3D(in_channels=int(model_meta.get("in_channels", 1)),
                       base_channels=int(model_meta.get("base_channels", 12)))
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        post = metadata.get("postprocessing", {})
        return cls(model=model, threshold=float(post.get("threshold", 0.5)), device=device,
                   keep_center_component=bool(post.get("keep_center_component", True)))

    @torch.no_grad()
    def predict_patch_probabilities(self, patch_hu: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(normalize_hu(patch_hu)[None, None]).to(self.device)
        return torch.sigmoid(self.model(x))[0, 0].detach().cpu().numpy().astype(np.float32)

    def segment_patch(self, patch_hu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probabilities = self.predict_patch_probabilities(patch_hu)
        mask = probabilities >= self.threshold
        if self.keep_center_component:
            mask = center_component(mask)
        return mask, probabilities

    def segment_volume_at_center(self, volume_xyz: np.ndarray, center_xyz: Tuple[float, float, float],
                                 spacing_xyz: Tuple[float, float, float],
                                 patch_size: int = DEFAULT_PATCH_SIZE) -> Dict[str, Any]:
        patch = extract_isotropic_patch(volume_xyz, center_xyz, spacing_xyz, patch_size=patch_size)
        patch_mask, probabilities = self.segment_patch(patch)
        native_mask = patch_mask_to_native(patch_mask, tuple(volume_xyz.shape), center_xyz, spacing_xyz)
        return {"patch_hu": patch, "patch_mask": patch_mask, "patch_probabilities": probabilities,
                "native_mask": native_mask}
