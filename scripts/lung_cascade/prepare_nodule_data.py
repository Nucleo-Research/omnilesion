#!/usr/bin/env python3
"""Build the training cache of the lung-nodule cascade from the nodule cohorts of the training set.

Cohorts: Chest_LIDC-IDRI, Chest_LNDb and Chest_LUNA25 (the challenge's three nodule cohorts), restricted to the
training cases of the split so that no held-out case is used. Per case (image and label reoriented to RAS):

* HU lung mask (omnilesion.lung_cascade.lung_mask), lung bounding box + 10 mm (whole image if no lung is found);
* DETECTOR sample: the crop resampled to 1 mm isotropic (linear), windowed to [-1024, 300] HU as uint8 ``.npy``, plus
  a JSON with every reference component as an xyzxyz box in 1 mm crop voxels (minimum side 3 mm), its volume and
  centroid;
* SEGMENTOR samples: one 81^3 patch at 1 mm per reference component, centred on its centroid, sampled from the
  native grid exactly as at inference (HU int16 + mask uint8, ``.npz``).

Cases with an empty label are skipped. Output: ``<out>/det/<case>.npy|json``, ``<out>/seg/<case>_<k>.npz``,
``<out>/manifest.jsonl`` (resumable). About 28 MB per case; the submission used all 4,967 training cases of the three
cohorts.

    python scripts/lung_cascade/prepare_nodule_data.py --raw $nnUNet_raw/Dataset501_OmniLesion \
        --split data/split_train_sentinel.json --out /fast_disk/nodule_cache --workers 14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omnilesion.lung_cascade.lung_mask import MARGIN_MM, WINDOW_HU, lung_mask  # noqa: E402

COHORTS = ("Chest_LUNA25", "Chest_LIDC-IDRI", "Chest_LNDb")
SEG_PATCH = 81
RAW: Path


def iso_patch(vol: np.ndarray, center, spacing, size: int, order: int, cval: float) -> np.ndarray:
    off = np.arange(size, dtype=np.float32) - (size - 1) / 2.0  # 1 mm steps
    xs = center[0] + off / spacing[0]
    ys = center[1] + off / spacing[1]
    zs = center[2] + off / spacing[2]
    grid = np.meshgrid(xs, ys, zs, indexing="ij")
    return ndimage.map_coordinates(vol, grid, order=order, mode="constant", cval=cval)


def process(case: str, out: Path) -> dict:
    t0 = time.time()
    img = nib.as_closest_canonical(nib.load(str(RAW / "imagesTr" / f"{case}_0000.nii.gz")))
    lab = nib.as_closest_canonical(nib.load(str(RAW / "labelsTr" / f"{case}.nii.gz")))
    hu = np.asarray(img.dataobj).astype(np.float32)
    gt = np.asarray(lab.dataobj) > 0
    if gt.shape != hu.shape:
        return {"case": case, "status": "shape_mismatch", "img": list(hu.shape), "lab": list(gt.shape)}
    sp = np.array([float(z) for z in img.header.get_zooms()[:3]], dtype=np.float64)
    voxel_ml = float(np.prod(sp)) / 1000.0
    comp_lab, n = ndimage.label(gt)
    if n == 0:
        return {"case": case, "status": "empty_label"}
    lm = lung_mask(hu, sp)
    if lm.any():
        idx = np.argwhere(lm)
        lo, hi = idx.min(0), idx.max(0) + 1
        m = np.ceil(MARGIN_MM / sp).astype(int)
        lo, hi = np.maximum(lo - m, 0), np.minimum(hi + m, hu.shape)
        lung_found = True
    else:
        lo, hi, lung_found = np.zeros(3, int), np.array(hu.shape), False
    lung_ml = float(lm.sum()) * voxel_ml
    crop = hu[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    out_shape = np.maximum(np.round(np.array(crop.shape) * sp).astype(int), 1)
    iso = ndimage.zoom(crop, out_shape / np.array(crop.shape), order=1, mode="nearest")
    u8 = (np.clip((iso - WINDOW_HU[0]) / (WINDOW_HU[1] - WINDOW_HU[0]), 0, 1) * 255.0).astype(np.uint8)
    objs = ndimage.find_objects(comp_lab)
    boxes, comps, dropped = [], [], 0
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        vox = np.argwhere(comp_lab == k)
        cen = vox.mean(0)
        vol_ml = len(vox) * voxel_ml
        b0 = (np.array([s.start for s in sl]) - lo) * sp
        b1 = (np.array([s.stop for s in sl]) - lo) * sp
        pad = np.maximum(3.0 - (b1 - b0), 0) / 2
        b0, b1 = b0 - pad, b1 + pad
        c_crop = (cen - lo) * sp
        inside = bool(np.all(c_crop >= 0) and np.all(c_crop < out_shape))
        if not inside:
            dropped += 1
        comps.append({"k": k, "centroid_native": [float(v) for v in cen], "centroid_crop_mm": [float(v) for v in c_crop],
                      "vol_ml": float(vol_ml), "n_vox": int(len(vox)), "in_crop": inside})
        if inside:
            boxes.append([float(v) for v in np.concatenate([np.maximum(b0, 0), np.minimum(b1, out_shape)])])
    np.save(out / "det" / f"{case}.npy", u8)
    meta = {"case": case, "cohort": case.rsplit("_", 1)[0], "native_shape": list(hu.shape),
            "native_spacing": [float(v) for v in sp], "crop_lo": [int(v) for v in lo], "crop_hi": [int(v) for v in hi],
            "crop_shape_1mm": [int(v) for v in out_shape], "lung_found": lung_found, "lung_ml": lung_ml,
            "boxes_xyzxyz": boxes, "labels": [0] * len(boxes), "components": comps,
            "dropped_outside_crop": dropped, "window_hu": list(WINDOW_HU)}
    json.dump(meta, (out / "det" / f"{case}.json").open("w"))
    n_seg = 0
    for c in comps:
        cen = c["centroid_native"]
        p_hu = iso_patch(hu, cen, sp, SEG_PATCH, order=1, cval=-1000.0)
        p_m = iso_patch((comp_lab == c["k"]).astype(np.float32), cen, sp, SEG_PATCH, order=0, cval=0.0) > 0.5
        np.savez_compressed(out / "seg" / f"{case}_{c['k']}.npz", hu=np.round(p_hu).astype(np.int16),
                            mask=p_m.astype(np.uint8), vol_ml=np.float32(c["vol_ml"]), spacing=sp.astype(np.float32))
        n_seg += 1
    return {"case": case, "status": "ok", "n_comp": n, "n_boxes": len(boxes), "n_seg": n_seg, "dropped": dropped,
            "lung_found": lung_found, "lung_ml": round(lung_ml), "shape_1mm": [int(v) for v in out_shape],
            "mb": round(u8.nbytes / 1e6, 1), "sec": round(time.time() - t0, 1)}


def _work(args):
    case, out = args
    try:
        return process(case, Path(out))
    except Exception as e:  # noqa: BLE001
        return {"case": case, "status": "error", "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-1500:]}


def main() -> None:
    global RAW
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True, help="nnU-Net raw dataset folder (imagesTr, labelsTr)")
    ap.add_argument("--split", type=Path, required=True, help="split JSON; only its 'train' cases are used")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n to split the work over machines")
    ap.add_argument("--cohorts", default=",".join(COHORTS))
    a = ap.parse_args()
    RAW = a.raw
    out = a.out
    (out / "det").mkdir(parents=True, exist_ok=True)
    (out / "seg").mkdir(parents=True, exist_ok=True)
    train = json.load(a.split.open())["train"]
    cohorts = tuple(a.cohorts.split(","))
    cases = sorted(c for c in train if c.rsplit("_", 1)[0] in cohorts)
    i, n = (int(x) for x in a.shard.split("/"))
    cases = cases[i::n]
    if a.limit:
        cases = cases[: a.limit]
    done = {json.loads(l)["case"] for l in (out / "manifest.jsonl").open()} if (out / "manifest.jsonl").exists() else set()
    todo = [c for c in cases if c not in done]
    print(f"{len(cases)} cases in shard {a.shard}, {len(done)} done, {len(todo)} to do; out={out}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool, (out / "manifest.jsonl").open("a") as mf:
        for j, r in enumerate(pool.imap_unordered(_work, [(c, str(out)) for c in todo]), 1):
            mf.write(json.dumps(r) + "\n")
            mf.flush()
            if r["status"] != "ok" or j % 50 == 0 or j <= 3:
                print(f"[{j}/{len(todo)}] {time.time() - t0:.0f}s {json.dumps({k: v for k, v in r.items() if k != 'tb'})}",
                      flush=True)
                if r["status"] == "error":
                    print(r["tb"], flush=True)
    print("PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
