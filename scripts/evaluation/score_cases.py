#!/usr/bin/env python3
"""Case-level DSC and NSD with the challenge's rules.

Per case: an empty prediction against an empty reference scores 1/1; exactly one empty scores 0/0; a prediction with
values above 1 is invalid and scores 0/0; otherwise DSC, and NSD at 2 mm tolerance, set to 0 when DSC < 0.2. Surface
distances come from the surface-distance package (DeepMind, Apache-2.0):
``pip install git+https://github.com/deepmind/surface-distance.git``.

    python scripts/evaluation/score_cases.py --pred masks/ --gt labels/ --out-csv scores.csv --out-json summary.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


def score_one(payload: tuple[str, str, str, float]) -> tuple[str, float, float, bool]:
    from surface_distance import compute_dice_coefficient, compute_surface_dice_at_tolerance, compute_surface_distances

    name, pred_dir, gt_dir, tolerance_mm = payload
    gt_nii = nib.load(str(Path(gt_dir) / name))
    spacing = gt_nii.header.get_zooms()[:3]
    gt = np.asarray(gt_nii.dataobj, dtype=np.uint8)
    pred = np.asarray(nib.load(str(Path(pred_dir) / name)).dataobj, dtype=np.uint8)
    invalid = bool(pred.max() > 1)
    gt_sum, pred_sum = int(gt.sum()), int(pred.sum())
    if invalid:
        dsc = nsd = 0.0
    elif gt_sum == 0 and pred_sum == 0:
        dsc = nsd = 1.0
    elif gt_sum == 0 or pred_sum == 0:
        dsc = nsd = 0.0
    else:
        dsc = float(compute_dice_coefficient(gt > 0, pred > 0))
        if dsc < 0.2:
            nsd = 0.0
        else:
            distances = compute_surface_distances(gt > 0, pred > 0, spacing)
            nsd = float(compute_surface_dice_at_tolerance(distances, tolerance_mm))
    return name, round(dsc, 4), round(nsd, 4), invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--tolerance-mm", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    names = sorted(p.name for p in arguments.pred.glob("*.nii.gz"))
    if not names:
        raise SystemExit(f"no .nii.gz predictions in {arguments.pred}")
    missing = [n for n in names if not (arguments.gt / n).exists()]
    if missing:
        raise SystemExit(f"missing ground truth for {len(missing)} predictions, e.g. {missing[:5]}")
    payloads = [(n, str(arguments.pred), str(arguments.gt), arguments.tolerance_mm) for n in names]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(score_one, payloads, chunksize=1))
    arguments.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out_csv.open("w") as handle:
        handle.write("Name,Lesion_DSC,Lesion_NSD\n")
        for name, dsc, nsd, _ in results:
            handle.write(f"{name},{dsc},{nsd}\n")
    summary = {"cases": len(results),
               "Lesion_DSC_mean": float(np.mean([r[1] for r in results])),
               "Lesion_NSD_mean": float(np.mean([r[2] for r in results])),
               "invalid_predictions": [r[0] for r in results if r[3]]}
    arguments.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
