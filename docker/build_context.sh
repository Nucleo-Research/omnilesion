#!/usr/bin/env bash
# Assemble the build context and build the image.
#   docker/build_context.sh <nnunet model folder> <lung weights folder> [image tag]
# <nnunet model folder> = $nnUNet_results/Dataset501_OmniLesion/nnUNetTrainerOmniLesion__nnUNetPlans_OmniLesion__3d_fullres
#                         (plans.json, dataset.json, fold_0/checkpoint_final.pth)
# <lung weights folder> = a folder holding detector.pt (a ckpt_*.pt of train_detector.py), segmentor_state_dict.pt and
#                         segmentor_manifest.json (from train_segmentor.py)
set -Eeuo pipefail
MODEL=$1; LUNG=$2; TAG=${3:-omnilesion:latest}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
CTX=$(mktemp -d); trap 'rm -rf "$CTX"' EXIT

cp -R "$ROOT/omnilesion" "$CTX/omnilesion"
find "$CTX/omnilesion" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$CTX/model/fold_0" "$CTX/lung"
cp "$MODEL/plans.json" "$MODEL/dataset.json" "$CTX/model/"
cp "$MODEL"/fold_*/checkpoint_final.pth "$CTX/model/fold_0/checkpoint_final.pth"
cp "$LUNG/detector.pt" "$LUNG/segmentor_state_dict.pt" "$LUNG/segmentor_manifest.json" "$CTX/lung/"
cp "$HERE/Dockerfile" "$HERE/predict_flare.py" "$HERE/predict.sh" "$CTX/"

docker build -t "$TAG" "$CTX"
echo "built $TAG; save with:  docker save $TAG | gzip -1 > omnilesion.tar.gz"
