#!/bin/sh
# Entry point invoked by the harness as `sh predict.sh`. All logic is in predict_flare.py.
set -u
echo "OmniLesion inference starting at $(date -u +%FT%TZ)"
echo "  binarise=${OMNILESION_BINARISE:-0.20} prune=${OMNILESION_PRUNE:-0.98} tta=${OMNILESION_TTA:-1} step=${OMNILESION_STEP_SIZE:-0.5} empty_rule=${OMNILESION_EMPTY_TOTAL_ML:-0.5}mL lung_cascade=${OMNILESION_LUNG_CASCADE:-1} tta_max_mvox=${OMNILESION_TTA_MAX_MVOX:-20} cut=${OMNILESION_LUNG_CUT:-0.98} gate=${OMNILESION_LUNG_GATE_ML:-3500}mL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  no GPU visible"
python /workspace/predict_flare.py
status=$?
echo "inference finished at $(date -u +%FT%TZ) with status ${status}"
echo "  wrote $(find /workspace/outputs -name '*.nii.gz' 2>/dev/null | wc -l) mask(s)"
exit ${status}
