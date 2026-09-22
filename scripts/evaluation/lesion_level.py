#!/usr/bin/env python3
"""Lesion-level detection statistics on completely annotated scans.

A lesion is a 26-connected component of the reference. It is detected when any predicted component shares a voxel
with it (IoU >= 0.1 is also reported); a predicted component touching no reference lesion is a false positive.
Recall is the detected share of reference lesions, precision the matched share of predicted components, and DSC on
detected lesions is computed between each detected lesion and the union of predicted components touching it, on a
crop around them, without the case-level zeroing rule. Everything is reported overall and by lesion volume band
(<1, 1-5, 5-20, >20 mL). Only meaningful on sets whose every lesion is annotated.

    python scripts/evaluation/lesion_level.py --pred masks/ --gt labels/ --out lesions.json [--per-lesion lesions.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

STRUCTURE = np.ones((3, 3, 3), dtype=bool)
BANDS = [(0.0, 1.0, "<1 mL"), (1.0, 5.0, "1-5 mL"), (5.0, 20.0, "5-20 mL"), (20.0, np.inf, ">20 mL")]


def band_of(volume_ml: float) -> str:
    for low, high, label in BANDS:
        if low <= volume_ml < high:
            return label
    return BANDS[-1][2]


def crop_dice(truth: np.ndarray, pred: np.ndarray) -> float:
    idx = np.where(truth | pred)
    crop = tuple(slice(max(int(i.min()) - 4, 0), int(i.max()) + 5) for i in idx)
    t, p = truth[crop], pred[crop]
    if not p.any():
        return 0.0
    return float(2.0 * (t & p).sum() / (t.sum() + p.sum()))


def analyse_case(name: str, pred_dir: Path, gt_dir: Path) -> tuple[list[dict], dict]:
    gt_nii = nib.load(str(gt_dir / name))
    voxel_ml = float(np.prod(gt_nii.header.get_zooms()[:3])) / 1000.0
    gt = np.asarray(gt_nii.dataobj) > 0
    pred = np.asarray(nib.load(str(pred_dir / name)).dataobj) > 0
    gt_lab, n_gt = ndimage.label(gt, structure=STRUCTURE)
    pr_lab, n_pr = ndimage.label(pred, structure=STRUCTURE)
    lesions = []
    matched_pred = set()
    for k in range(1, n_gt + 1):
        lesion = gt_lab == k
        touching = set(np.unique(pr_lab[lesion])) - {0}
        matched_pred |= touching
        union = np.isin(pr_lab, list(touching)) if touching else np.zeros_like(lesion)
        inter = float((lesion & union).sum())
        iou = inter / float((lesion | union).sum()) if touching else 0.0
        lesions.append({"case": name, "lesion": k, "volume_ml": float(lesion.sum()) * voxel_ml,
                        "detected": bool(touching), "detected_iou01": iou >= 0.1,
                        "dsc": crop_dice(lesion, union) if touching else 0.0})
    fp_components = [k for k in range(1, n_pr + 1) if k not in matched_pred]
    fp_ml = [float((pr_lab == k).sum()) * voxel_ml for k in fp_components]
    return lesions, {"case": name, "n_pred_components": n_pr, "n_false_positive": len(fp_components),
                     "false_positive_ml": fp_ml}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-lesion", type=Path, default=None, help="optional CSV with one row per reference lesion")
    arguments = parser.parse_args()
    names = sorted(p.name for p in arguments.pred.glob("*.nii.gz") if (arguments.gt / p.name).exists())
    all_lesions, case_rows = [], []
    for name in names:
        lesions, case_row = analyse_case(name, arguments.pred, arguments.gt)
        all_lesions += lesions
        case_rows.append(case_row)

    def summary(rows: list[dict]) -> dict:
        det = [r for r in rows if r["detected"]]
        return {"lesions": len(rows), "recall": len(det) / max(len(rows), 1),
                "recall_iou01": sum(r["detected_iou01"] for r in rows) / max(len(rows), 1),
                "dsc_detected_mean": float(np.mean([r["dsc"] for r in det])) if det else None}

    by_band = defaultdict(list)
    for r in all_lesions:
        by_band[band_of(r["volume_ml"])].append(r)
    n_pred = sum(c["n_pred_components"] for c in case_rows)
    n_fp = sum(c["n_false_positive"] for c in case_rows)
    fp_ml = [v for c in case_rows for v in c["false_positive_ml"]]
    result = {"cases": len(names), "overall": summary(all_lesions),
              "by_volume": {label: summary(by_band[label]) for _, _, label in BANDS},
              "precision": (n_pred - n_fp) / max(n_pred, 1), "predicted_components": n_pred,
              "false_positives": n_fp, "false_positives_per_case": n_fp / max(len(names), 1),
              "false_positive_median_ml": float(np.median(fp_ml)) if fp_ml else None,
              "false_positive_share_below_1ml": float(np.mean([v < 1.0 for v in fp_ml])) if fp_ml else None}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, indent=2) + "\n")
    if arguments.per_lesion:
        with arguments.per_lesion.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_lesions[0].keys()) if all_lesions else ["case"])
            writer.writeheader()
            writer.writerows(all_lesions)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
