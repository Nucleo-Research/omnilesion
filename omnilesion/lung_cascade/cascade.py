"""Lung-nodule cascade at inference: gate, lung-box crop at 1 mm, RetinaNet proposals, 65^3 U-Net masks, union.

Works on the nnU-Net SimpleITK array order (z, y, x) by transposing to (x, y, z) with the spacing reversed. Both
networks were trained with random flips on every axis, so the orientation of the input frame does not matter.
Environment variables (defaults are the submitted operating point):

    OMNILESION_LUNG_DIR       folder with detector.pt, segmentor_state_dict.pt, segmentor_manifest.json
    OMNILESION_LUNG_CUT       0.98   detector confidence needed for a proposal to contribute its mask
    OMNILESION_LUNG_GATE_ML   3500   minimum lung volume (mL) for the cascade to run at all
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from .detector import PATCH, build_detector
from .lung_mask import gate_and_box, window_to_unit
from .nodule_unet import NoduleUNetSegmenter

LUNG_DIR = Path(os.environ.get("OMNILESION_LUNG_DIR", "/opt/lung"))
CUT = float(os.environ.get("OMNILESION_LUNG_CUT", "0.98"))
MIN_LUNG_ML = float(os.environ.get("OMNILESION_LUNG_GATE_ML", "3500"))


class LungCascade:
    def __init__(self, device: torch.device, lung_dir: Path = LUNG_DIR, cut: float = CUT,
                 min_lung_ml: float = MIN_LUNG_ML):
        self.device, self.cut, self.min_lung_ml = device, float(cut), float(min_lung_ml)
        checkpoint = torch.load(str(lung_dir / "detector.pt"), map_location="cpu", weights_only=False)
        self.det = build_detector(device, PATCH, training=False, sw_batch_size=4)
        self.det.network.load_state_dict(checkpoint["model"])
        self.det.eval()
        self.seg = NoduleUNetSegmenter.from_checkpoint(lung_dir / "segmentor_state_dict.pt",
                                                       lung_dir / "segmentor_manifest.json", device=device)
        print(f"[lung] detector iteration {checkpoint.get('iter')} cut {self.cut} gate {self.min_lung_ml:.0f} mL",
              flush=True)

    @torch.no_grad()
    def __call__(self, hu_zyx: np.ndarray, spacing_zyx, base_zyx: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return the base mask unioned with the cascade's nodule masks (z, y, x, uint8) and timing/count info."""
        t0 = time.time()
        hu = np.ascontiguousarray(hu_zyx.transpose(2, 1, 0)).astype(np.float32)
        sp = np.array(spacing_zyx[::-1], dtype=np.float64)
        lung_ml, lo, hi = gate_and_box(hu, sp)
        info = {"lung_ml": round(lung_ml), "gate_s": round(time.time() - t0, 2),
                "gated_in": bool(lung_ml >= self.min_lung_ml and lo is not None)}
        if not info["gated_in"]:
            return base_zyx, info
        t1 = time.time()
        crop = hu[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        out_shape = np.maximum(np.round(np.array(crop.shape) * sp).astype(int), 1)
        iso = ndimage.zoom(crop, out_shape / np.array(crop.shape), order=1, mode="nearest")
        img = window_to_unit(iso)
        info["zoom_s"] = round(time.time() - t1, 2)
        t2 = time.time()
        with torch.autocast("cuda", enabled=self.device.type == "cuda"):
            pred = self.det([torch.from_numpy(img)[None].to(self.device)], use_inferer=True)[0]
        boxes = pred["box"].float().cpu().numpy().reshape(-1, 6)
        scores = pred["label_scores"].float().cpu().numpy().ravel()
        info["det_s"] = round(time.time() - t2, 2)
        t3 = time.time()
        base = np.ascontiguousarray(base_zyx.transpose(2, 1, 0)).astype(bool)
        fused = base.copy()
        added = n_candidates = 0
        for box, score in zip(boxes, scores):
            if score < self.cut:
                continue
            n_candidates += 1
            # boxes are [start, stop) in 1 mm crop voxels: the index-space centre is the midpoint minus 0.5
            centre_crop = (box[:3] + box[3:]) / 2 - 0.5
            centre_native = lo + centre_crop / sp
            mask = self.seg.segment_volume_at_center(hu, tuple(float(x) for x in centre_native),
                                                     tuple(float(x) for x in sp))["native_mask"]
            if mask.any():
                fused |= mask
                added += 1
        info.update({"seg_s": round(time.time() - t3, 2), "n_candidates": n_candidates, "n_added": added,
                     "added_voxels": int((fused & ~base).sum()), "cascade_s": round(time.time() - t0, 2)})
        return np.ascontiguousarray(fused.transpose(2, 1, 0)).astype(np.uint8), info
