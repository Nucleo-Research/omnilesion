#!/usr/bin/env python3
"""Derive the OmniLesion training plan from the plan that nnU-Net generated for the dataset.

The derived plan reuses the preprocessed data of the base plan (same ``data_identifier``, so preprocessing runs once)
and replaces the 3d_fullres configuration with the submitted one: 3 x 2 x 2 mm, 96 x 128 x 128 patches, batch 2,
batch Dice, CT normalisation, nnU-Net's default resampling, and a residual-encoder U-Net with six stages of
32/64/128/256/320/320 features and 1/3/4/6/6/6 residual blocks (one convolution per decoder stage). Everything else
in the plan (fingerprint, transposition, intensity statistics) is kept from the base plan.

    python scripts/make_resenc_plans.py --dataset-id 501 --base-plans nnUNetPlans_OmniLesionBase \
        --out-plans nnUNetPlans_OmniLesion

The exact plan of the submitted model is data/plans_omnilesion.json; ``--reference`` compares against it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

ARCHITECTURE = {
    "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
    "arch_kwargs": {
        "n_stages": 6,
        "features_per_stage": [32, 64, 128, 256, 320, 320],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * 6,
        "strides": [[1, 1, 1]] + [[2, 2, 2]] * 5,
        "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
        "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True},
    },
    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
}
CONFIGURATION = {
    "spacing": [3.0, 2.0, 2.0],
    "patch_size": [96, 128, 128],
    "batch_size": 2,
    "batch_dice": True,
    "normalization_schemes": ["CTNormalization"],
    "use_mask_for_norm": [False],
    "resampling_fn_data": "resample_data_or_seg_to_shape",
    "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
    "resampling_fn_seg": "resample_data_or_seg_to_shape",
    "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
    "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
    "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
    "architecture": ARCHITECTURE,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--preprocessed", type=Path, default=Path(os.environ.get("nnUNet_preprocessed", "")))
    parser.add_argument("--base-plans", default="nnUNetPlans_OmniLesionBase")
    parser.add_argument("--out-plans", default="nnUNetPlans_OmniLesion")
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "plans_omnilesion.json")
    arguments = parser.parse_args()
    folder = next(arguments.preprocessed.glob(f"Dataset{arguments.dataset_id:03d}_*"))
    base = json.loads((folder / f"{arguments.base_plans}.json").read_text())

    plan = copy.deepcopy(base)
    plan["plans_name"] = arguments.out_plans
    cfg = plan["configurations"]["3d_fullres"]
    for key, value in CONFIGURATION.items():
        cfg[key] = copy.deepcopy(value)
    cfg["data_identifier"] = base["configurations"]["3d_fullres"].get("data_identifier",
                                                                       f"{arguments.base_plans}_3d_fullres")
    out = folder / f"{arguments.out_plans}.json"
    out.write_text(json.dumps(plan, indent=4) + "\n")
    print(f"wrote {out} (data_identifier {cfg['data_identifier']})")

    fg = plan["foreground_intensity_properties_per_channel"]["0"]
    print(f"foreground CT statistics of this corpus: mean {fg['mean']} std {fg['std']} "
          f"(clip {fg['percentile_00_5']} .. {fg['percentile_99_5']})")
    if arguments.reference.is_file():
        ref = json.loads(arguments.reference.read_text())["foreground_intensity_properties_per_channel"]["0"]
        if abs(ref["mean"] - fg["mean"]) > 1e-3 or abs(ref["std"] - fg["std"]) > 1e-3:
            print("they differ from the submitted model's; export before training:\n"
                  f"    export OMNILESION_FOREGROUND_MEAN_HU={fg['mean']}\n"
                  f"    export OMNILESION_FOREGROUND_STD_HU={fg['std']}")
        else:
            print("they match the submitted model's plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
