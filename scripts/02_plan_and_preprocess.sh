#!/usr/bin/env bash
# Fingerprint, plan and preprocess the dataset at 3 x 2 x 2 mm, then derive the OmniLesion plan.
# Requires nnUNet_raw / nnUNet_preprocessed / nnUNet_results to be exported (see README).
set -Eeuo pipefail
DATASET_ID=${DATASET_ID:-501}
WORKERS=${WORKERS:-16}
HERE=$(cd "$(dirname "$0")" && pwd)

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity --no_pp --clean -npfp "$WORKERS" \
  -overwrite_target_spacing 3.0 2.0 2.0 -overwrite_plans_name nnUNetPlans_OmniLesionBase -c 3d_fullres --verbose

python "$HERE/make_resenc_plans.py" --dataset-id "$DATASET_ID" \
  --base-plans nnUNetPlans_OmniLesionBase --out-plans nnUNetPlans_OmniLesion

# Preprocess once with the base plan; the derived plan shares its data_identifier.
nnUNetv2_preprocess -d "$DATASET_ID" -plans_name nnUNetPlans_OmniLesionBase -c 3d_fullres -np "$WORKERS" --verbose
