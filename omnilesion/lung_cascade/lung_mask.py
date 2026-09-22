"""Lung mask and lung gate from Hounsfield units alone (no anatomical model), shared by training and inference."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

MARGIN_MM = 10.0            # lung bounding box margin
WINDOW_HU = (-1024.0, 300.0)  # detector input window
GATE_SUBSAMPLE = np.array([4, 4, 2])  # subsampling of the HU volume for the gate (x, y, z)


def lung_mask(hu: np.ndarray, spacing) -> np.ndarray:
    """Dilated lung parenchyma: body = largest component above -500 HU, filled slice by slice along the last axis;
    air = voxels below -320 HU inside the body; keep air components above 100 mL; dilate by 5 mm."""
    voxel_ml = float(np.prod(spacing)) / 1000.0
    body = hu > -500
    lab, n = ndimage.label(body)
    if n > 1:
        sizes = ndimage.sum(body, lab, np.arange(1, n + 1))
        body = lab == (1 + int(np.argmax(sizes)))
    filled = np.zeros_like(body)
    for k in range(body.shape[2]):
        filled[:, :, k] = ndimage.binary_fill_holes(body[:, :, k])
    air = (hu < -320) & filled
    lab, n = ndimage.label(air)
    if not n:
        return np.zeros_like(hu, dtype=bool)
    sizes = ndimage.sum(air, lab, np.arange(1, n + 1)) * voxel_ml
    keep = np.arange(1, n + 1)[sizes > 100.0]
    if not len(keep):
        return np.zeros_like(hu, dtype=bool)
    lungs = np.isin(lab, keep)
    iterations = max(1, int(round(5.0 / float(min(spacing)))))
    return ndimage.binary_dilation(lungs, iterations=iterations)


def gate_and_box(hu_xyz: np.ndarray, spacing_xyz: np.ndarray, subsample: np.ndarray = GATE_SUBSAMPLE,
                 margin_mm: float = MARGIN_MM):
    """Lung volume in mL of the two largest air components on a subsampled grid, and the lung bounding box (native
    voxel indices, padded by ``margin_mm``). Returns ``(volume_ml, lo, hi)``; ``lo``/``hi`` are None without lung."""
    sub = hu_xyz[:: subsample[0], :: subsample[1], :: subsample[2]]
    lm = lung_mask(sub, spacing_xyz * subsample)
    lab, n = ndimage.label(lm)
    if n == 0:
        return 0.0, None, None
    if n > 2:
        sizes = ndimage.sum(lm, lab, np.arange(1, n + 1))
        core = np.isin(lab, np.argsort(sizes)[-2:] + 1)
    else:
        core = lm
    vol_ml = float(core.sum()) * float(np.prod(spacing_xyz * subsample)) / 1000.0
    idx = np.argwhere(core) * subsample
    m = np.ceil(margin_mm / spacing_xyz).astype(int)
    lo = np.maximum(idx.min(0) - m, 0)
    hi = np.minimum(idx.max(0) + subsample + m, hu_xyz.shape)
    return vol_ml, lo, hi


def window_to_unit(iso_hu: np.ndarray, window=WINDOW_HU) -> np.ndarray:
    """Clip to the detector window and map to [0, 1] (float32)."""
    return np.clip((iso_hu - window[0]) / (window[1] - window[0]), 0, 1).astype(np.float32)
