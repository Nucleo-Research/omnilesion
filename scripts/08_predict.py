#!/usr/bin/env python3
"""Segment a folder of CT scans with the full pipeline (main model, post-processing, lung-nodule cascade).

    python scripts/08_predict.py --inputs /data/images --outputs /data/masks \
        --model $nnUNet_results/Dataset501_OmniLesion/nnUNetTrainerOmniLesion__nnUNetPlans_OmniLesion__3d_fullres \
        --lung-dir /data/lung_weights

Inputs are ``<case>_0000.nii.gz`` (or any ``.nii.gz``); outputs are ``<case>.nii.gz`` uint8 masks. All operating-point
knobs are environment variables documented in omnilesion/inference.py; ``--no-cascade`` runs the main model alone.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="nnU-Net model folder (plans.json, fold_*/)")
    parser.add_argument("--fold", default="0")
    parser.add_argument("--lung-dir", type=Path, default=None,
                        help="folder with detector.pt, segmentor_state_dict.pt, segmentor_manifest.json")
    parser.add_argument("--no-cascade", action="store_true")
    arguments = parser.parse_args()
    os.environ["OMNILESION_MODEL"] = str(arguments.model)
    os.environ["OMNILESION_FOLD"] = arguments.fold
    if arguments.no_cascade or arguments.lung_dir is None:
        os.environ["OMNILESION_LUNG_CASCADE"] = "0"
    else:
        os.environ["OMNILESION_LUNG_DIR"] = str(arguments.lung_dir)
    from omnilesion import inference  # noqa: E402  (reads the environment at import)

    return inference.run(arguments.inputs, arguments.outputs, arguments.model)


if __name__ == "__main__":
    raise SystemExit(main())
