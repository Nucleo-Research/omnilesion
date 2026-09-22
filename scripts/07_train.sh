#!/usr/bin/env bash
# Train the main model: 4000 epochs x 750 iterations (3.0M updates) on one GPU, ~54 h on an H100.
# The trainer must have been installed into nnU-Net first: python -m omnilesion.nnunet_trainer.install
set -Eeuo pipefail
DATASET_ID=${DATASET_ID:-501}
FOLD=${FOLD:-0}                       # only names the output folder; the split comes from OMNILESION_SPLIT_JSON
PREPROCESSED_DIR=$(ls -d "${nnUNet_preprocessed:?export nnUNet_preprocessed}"/Dataset$(printf %03d "$DATASET_ID")_*)

export OMNILESION_SPLIT_JSON=${OMNILESION_SPLIT_JSON:-$PREPROCESSED_DIR/split_train_sentinel.json}
export OMNILESION_CASE_WEIGHTS_JSON=${OMNILESION_CASE_WEIGHTS_JSON:-$PREPROCESSED_DIR/case_weights_alpha07.json}
export OMNILESION_EPOCHS=${OMNILESION_EPOCHS:-4000}
export OMNILESION_ITERS_PER_EPOCH=${OMNILESION_ITERS_PER_EPOCH:-750}
export OMNILESION_LR=${OMNILESION_LR:-0.01}
export OMNILESION_POLY_EXPONENT=${OMNILESION_POLY_EXPONENT:-2.0}
export OMNILESION_COMPONENT_WEIGHT=${OMNILESION_COMPONENT_WEIGHT:-1.0}
export OMNILESION_MAX_COMPONENTS=${OMNILESION_MAX_COMPONENTS:-48}
export nnUNet_compile=${nnUNet_compile:-true}   # torch.compile of the network, ~18 % faster on an H100

for f in "$OMNILESION_SPLIT_JSON" "$OMNILESION_CASE_WEIGHTS_JSON"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done
echo "epochs=$OMNILESION_EPOCHS iters/epoch=$OMNILESION_ITERS_PER_EPOCH lr=$OMNILESION_LR exponent=$OMNILESION_POLY_EXPONENT"
nnUNetv2_train "$DATASET_ID" 3d_fullres "$FOLD" -tr nnUNetTrainerOmniLesion -p nnUNetPlans_OmniLesion -num_gpus 1 "$@"
# Resume an interrupted run with:  scripts/07_train.sh --c
