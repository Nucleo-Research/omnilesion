#!/usr/bin/env python3
"""Relabel a trained nnU-Net model folder so that it loads with this repository's names.

nnU-Net stores the trainer class name and a copy of the plan in the checkpoint, and the plan in plans.json; a model trained under other
names is renamed in place (a backup of the checkpoint is kept):

    python tools/relabel_release_weights.py --model <model folder> \
        --trainer nnUNetTrainerOmniLesion --plans nnUNetPlans_OmniLesion --dataset Dataset501_OmniLesion
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trainer", default="nnUNetTrainerOmniLesion")
    parser.add_argument("--plans", default="nnUNetPlans_OmniLesion")
    parser.add_argument("--dataset", default="Dataset501_OmniLesion")
    parser.add_argument("--data-identifier", default="nnUNetPlans_OmniLesionBase_3d_fullres",
                        help="preprocessed-data identifier written into the 3d_fullres configuration")
    arguments = parser.parse_args()

    plans_path = arguments.model / "plans.json"
    plans = json.loads(plans_path.read_text())
    plans["plans_name"], plans["dataset_name"] = arguments.plans, arguments.dataset
    plans["configurations"]["3d_fullres"]["data_identifier"] = arguments.data_identifier
    plans_path.write_text(json.dumps(plans, indent=4) + "\n")
    dataset_path = arguments.model / "dataset.json"
    if dataset_path.exists():
        dataset = json.loads(dataset_path.read_text())
        dataset["dataset_name"] = arguments.dataset
        dataset_path.write_text(json.dumps(dataset, indent=4) + "\n")
    for checkpoint_path in sorted(arguments.model.glob("fold_*/checkpoint_*.pth")):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        before = checkpoint.get("trainer_name")
        checkpoint["trainer_name"] = arguments.trainer
        init_args = checkpoint.get("init_args")
        if isinstance(init_args, dict) and "plans" in init_args:
            init_args["plans"]["plans_name"] = arguments.plans
            init_args["plans"]["dataset_name"] = arguments.dataset
            init_args["plans"]["configurations"]["3d_fullres"]["data_identifier"] = arguments.data_identifier
        backup = checkpoint_path.with_suffix(".pth.bak")
        if not backup.exists():
            shutil.copyfile(checkpoint_path, backup)
        torch.save(checkpoint, checkpoint_path)
        print(f"{checkpoint_path}: trainer_name {before} -> {arguments.trainer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
