#!/usr/bin/env python3
"""Share of lesion-free scans on which the pipeline returns an empty mask (the only outcome the metric rewards).

    python scripts/evaluation/lesion_free_rate.py --pred masks_of_lesion_free_scans/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    arguments = parser.parse_args()
    rows = []
    for path in sorted(arguments.pred.glob("*.nii.gz")):
        nii = nib.load(str(path))
        voxels = int((np.asarray(nii.dataobj) > 0).sum())
        rows.append({"case": path.name, "voxels": voxels,
                     "ml": voxels * float(np.prod(nii.header.get_zooms()[:3])) / 1000.0})
    empty = sum(r["voxels"] == 0 for r in rows)
    not_empty = [r["ml"] for r in rows if r["voxels"]]
    result = {"cases": len(rows), "empty": empty, "empty_share": empty / max(len(rows), 1),
              "median_ml_when_not_empty": float(np.median(not_empty)) if not_empty else None}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
