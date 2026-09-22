# OmniLesion inference image for the FLARE 2026 Task 1 harness.
#
# Harness contract: inputs and outputs are bind mounts, the working directory holds predict.sh, no network at run time:
#     docker container run --gpus "device=0" -m 28G --rm \
#         -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ \
#         omnilesion:latest /bin/bash -c "sh predict.sh"
#
# Build context (assembled by docker/build_context.sh): omnilesion/ (this package), model/ (plans.json, dataset.json,
# fold_0/checkpoint_final.pth), lung/ (detector.pt, segmentor_state_dict.pt, segmentor_manifest.json), predict_flare.py,
# predict.sh. torch comes from the base image; pip never touches it.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    nnUNet_raw=/opt/nnunet/raw nnUNet_preprocessed=/opt/nnunet/preprocessed nnUNet_results=/opt/nnunet/results \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# SimpleITK's shared objects need these at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Pinned so that a rebuild cannot silently change the numerics. nnU-Net and MONAI with --no-deps so pip leaves torch alone.
RUN python -m pip install --no-cache-dir \
        SimpleITK==2.4.1 nibabel==5.3.2 scipy==1.15.2 scikit-image==0.25.2 numpy==1.26.4 pandas==2.2.3 tqdm==4.67.1 \
        dynamic-network-architectures==0.4.1 acvl-utils==0.2.3 batchgenerators==0.25.1 batchgeneratorsv2==0.2.3 \
        graphviz==0.20.3 connected-components-3d==3.21.0 matplotlib==3.11.1 seaborn==0.13.2 einops==0.8.1 \
        blosc2==3.2.1 yacs==0.1.8 imagecodecs==2024.12.30 scikit-learn==1.6.1 tifffile==2025.3.13 requests==2.33.0 && \
    python -m pip install --no-cache-dir --no-deps nnunetv2==2.6.0 monai==1.6.0

# The OmniLesion package and its trainer (nnU-Net rebuilds the network from checkpoint['trainer_name']).
COPY omnilesion /workspace/omnilesion
RUN cd /workspace && python -m omnilesion.nnunet_trainer.install

# Trained main model in nnU-Net's layout, then the lung cascade weights and the entry point.
COPY model /opt/model
COPY lung /opt/lung
COPY predict_flare.py /workspace/predict_flare.py
COPY predict.sh /workspace/predict.sh
WORKDIR /workspace
RUN mkdir -p /workspace/inputs /workspace/outputs && chmod +x /workspace/predict.sh

# Fail at build time, not during the timed run, if the trainer or the cascade cannot be constructed.
RUN python -c "import os, sys, torch, nnunetv2; sys.path.insert(0, '/workspace'); \
from omnilesion.lung_cascade.cascade import LungCascade; \
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class; \
name = torch.load('/opt/model/fold_0/checkpoint_final.pth', map_location='cpu', weights_only=False)['trainer_name']; \
c = recursive_find_python_class(os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'), name, 'nnunetv2.training.nnUNetTrainer'); \
assert c is not None, name; print('trainer resolved:', c); LungCascade(torch.device('cpu')); print('lung cascade resolved')"

CMD ["/bin/bash", "-c", "sh predict.sh"]
