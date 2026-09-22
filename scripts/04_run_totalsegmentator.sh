#!/usr/bin/env bash
# Run TotalSegmentator (task "total", 117 classes, multilabel output) on every training image, for the organ-family
# attribution of scripts/05_attribute_lesions.py. Output: <out>/<case>.nii.gz in the image grid.
#   scripts/04_run_totalsegmentator.sh $nnUNet_raw/Dataset501_OmniLesion/imagesTr /data/totalseg
# Set FAST=1 to use TotalSegmentator's 3 mm model (much faster; adequate for organ-level attribution).
set -Eeuo pipefail
IMAGES=$1; OUT=$2; mkdir -p "$OUT"
for img in "$IMAGES"/*_0000.nii.gz; do
  case=$(basename "$img" _0000.nii.gz)
  [ -e "$OUT/$case.nii.gz" ] && continue
  TotalSegmentator -i "$img" -o "$OUT/$case.nii.gz" --ml --task total ${FAST:+--fast} >/dev/null
done
