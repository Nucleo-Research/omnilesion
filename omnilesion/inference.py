"""Whole pipeline at inference: nnU-Net prediction in-process, post-processing, lung-nodule cascade, NIfTI output.

Used by ``scripts/08_predict.py`` (offline, any folder) and by the container entry point ``docker/predict_flare.py``.
The operating point is fixed; every knob is an environment variable with the submitted value as default:

    OMNILESION_MODEL            trained model folder (plans.json, dataset.json, fold_*/checkpoint_final.pth)
    OMNILESION_FOLD             0          folds to load (comma separated)
    OMNILESION_CHECKPOINT       checkpoint_final.pth
    OMNILESION_STEP_SIZE        0.5        sliding-window step
    OMNILESION_TTA              1          mirroring test-time augmentation (all axes)
    OMNILESION_TTA_MAX_MVOX     20         mirroring is switched off above this many million voxels on the 3x2x2 grid
    OMNILESION_BINARISE         0.20       see omnilesion.postprocess
    OMNILESION_PRUNE            0.98
    OMNILESION_EMPTY_TOTAL_ML   0.5        0 disables the empty rule
    OMNILESION_EMPTY_MAX_COMPONENTS  2
    OMNILESION_LUNG_CASCADE     1          0 disables the cascade (see omnilesion.lung_cascade for its knobs)
    OMNILESION_LUNG_TIME_GUARD_S 25        the cascade is skipped when the main model already took longer than this
    OMNILESION_GPU_RESAMPLE     1          resample image and probabilities on the GPU (trilinear, chunked)

GPU resampling replaces nnU-Net's CPU resampling (cubic spline in, linear out) by torch trilinear interpolation. It is
not bit-identical to the CPU path; it was validated end to end on the held-out and hidden validation sets. When it
fails on a case the case is retried on the CPU path, and when that fails too an empty mask is written so that every
input gets an output.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .postprocess import postprocess

MODEL = Path(os.environ.get("OMNILESION_MODEL", "/opt/model"))
FOLDS = os.environ.get("OMNILESION_FOLD", "0")
CHECKPOINT = os.environ.get("OMNILESION_CHECKPOINT", "checkpoint_final.pth")
STEP = float(os.environ.get("OMNILESION_STEP_SIZE", "0.5"))
USE_MIRRORING = os.environ.get("OMNILESION_TTA", "1") != "0"
TTA_MAX_MVOX = float(os.environ.get("OMNILESION_TTA_MAX_MVOX", "20"))
BINARISE = float(os.environ.get("OMNILESION_BINARISE", "0.20"))
PRUNE = float(os.environ.get("OMNILESION_PRUNE", "0.98"))
EMPTY_TOTAL_ML = float(os.environ.get("OMNILESION_EMPTY_TOTAL_ML", "0.5"))
EMPTY_MAX_COMPONENTS = int(os.environ.get("OMNILESION_EMPTY_MAX_COMPONENTS", "2"))
LUNG_CASCADE = os.environ.get("OMNILESION_LUNG_CASCADE", "1") != "0"
LUNG_TIME_GUARD_S = float(os.environ.get("OMNILESION_LUNG_TIME_GUARD_S", "25"))
GPU_RESAMPLE = os.environ.get("OMNILESION_GPU_RESAMPLE", "1") != "0"


# ----- GPU resampling ------------------------------------------------------------------------------------------------

def _interp_chunked(data, new_shape, mode, device, chunk):
    """F.interpolate over the leading axis in chunks, accumulating on the CPU, so the GPU holds one chunk at a time."""
    import torch.nn.functional as F
    was_numpy = not isinstance(data, torch.Tensor)
    t = torch.from_numpy(data) if was_numpy else data
    orig_device = t.device
    out = []
    for i in range(0, t.shape[0], chunk):
        part = t[i:i + chunk].to(device).float()
        res = F.interpolate(part[None], tuple(int(v) for v in new_shape), mode=mode, antialias=False)[0]
        out.append(res.cpu())
        del part, res
    result = torch.cat(out, 0)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result.numpy() if was_numpy else result.to(orig_device)


def _resample_torch(data, new_shape, current_spacing, new_spacing, is_seg=False, num_threads=4, device=None,
                    memefficient_seg_resampling=False, force_separate_z=None, separate_z_anisotropy_threshold=None,
                    mode="linear", aniso_axis_mode="nearest-exact"):
    """nnU-Net's ``resample_torch_fornnunet`` for data (not segmentations), with chunked interpolation and the
    anisotropic axis taken as a scalar."""
    from einops import rearrange
    from nnunetv2.preprocessing.resampling.default_resampling import ANISO_THRESHOLD, determine_do_sep_z_and_axis

    assert not is_seg, "segmentation resampling is not routed through this function"
    if separate_z_anisotropy_threshold is None:
        separate_z_anisotropy_threshold = ANISO_THRESHOLD
    device = device or torch.device("cpu")
    assert data.ndim == 4, "data must be c, x, y, z"
    new_shape = [int(i) for i in new_shape]
    orig_shape = data.shape
    if all(i == j for i, j in zip(new_shape, data.shape[1:])):
        return data
    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)
    if not do_separate_z:
        return _interp_chunked(data, new_shape, "trilinear", device, chunk=1)
    axis = int(np.atleast_1d(axis)[0])
    was_numpy = isinstance(data, np.ndarray)
    t = torch.from_numpy(data) if was_numpy else data
    letters = "xyz"
    axis_letter = letters[axis]
    others_int = [i for i in range(3) if i != axis]
    others = [letters[i] for i in others_int]
    t = rearrange(t, f"c x y z -> (c {axis_letter}) {others[0]} {others[1]}")
    tmp_new_shape = [new_shape[i] for i in others_int]
    t = _interp_chunked(t, tmp_new_shape, "bilinear", device, chunk=64)
    t = rearrange(t, f"(c {axis_letter}) {others[0]} {others[1]} -> c x y z",
                  **{axis_letter: orig_shape[axis + 1], others[0]: tmp_new_shape[0], others[1]: tmp_new_shape[1]})
    t = _interp_chunked(t, new_shape, aniso_axis_mode, device, chunk=1)
    return t.numpy() if was_numpy else t


def set_resampling(configuration_manager, gpu: bool, device=None) -> None:
    """Switch the configuration between nnU-Net's CPU resampling (skimage) and GPU torch resampling."""
    import nnunetv2.preprocessing.resampling.resample_torch as rt
    rt.resample_torch_fornnunet = _resample_torch
    cfg = configuration_manager.configuration
    if gpu:
        for key in ("resampling_fn_data", "resampling_fn_probabilities"):
            cfg[key] = "resample_torch_fornnunet"
            cfg[key + "_kwargs"] = {"is_seg": False, "num_threads": int(os.environ.get("OMP_NUM_THREADS", "4")),
                                    "device": device, "force_separate_z": None}
    else:
        cfg["resampling_fn_data"] = "resample_data_or_seg_to_shape"
        cfg["resampling_fn_data_kwargs"] = {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None}
        cfg["resampling_fn_probabilities"] = "resample_data_or_seg_to_shape"
        cfg["resampling_fn_probabilities_kwargs"] = {"is_seg": False, "order": 1, "order_z": 0,
                                                     "force_separate_z": None}
    for name in ("resampling_fn_data", "resampling_fn_probabilities", "resampling_fn_seg"):
        prop = getattr(type(configuration_manager), name, None)
        fget = getattr(prop, "fget", None)
        if fget is not None and hasattr(fget, "cache_clear"):
            fget.cache_clear()


# ----- predictor -----------------------------------------------------------------------------------------------------

def build_predictor(model: Path = MODEL):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(tile_step_size=STEP, use_gaussian=True, use_mirroring=USE_MIRRORING,
                                perform_everything_on_device=device.type == "cuda", device=device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    folds = tuple(int(f) for f in FOLDS.split(",") if f.strip())
    predictor.initialize_from_trained_model_folder(str(model), use_folds=folds, checkpoint_name=CHECKPOINT)
    if GPU_RESAMPLE and device.type == "cuda":
        set_resampling(predictor.configuration_manager, gpu=True, device=device)
    print(f"[init] device={device} folds={folds} checkpoint={CHECKPOINT} lung_cascade={LUNG_CASCADE} "
          f"gpu_resample={GPU_RESAMPLE} mirroring={USE_MIRRORING} step={STEP} binarise={BINARISE} prune={PRUNE} "
          f"empty_if<={EMPTY_TOTAL_ML}mL&<={EMPTY_MAX_COMPONENTS}comp", flush=True)
    return predictor


def build_cascade():
    if not LUNG_CASCADE:
        return None
    try:
        from .lung_cascade.cascade import LungCascade
        return LungCascade(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception:
        traceback.print_exc()
        print("[lung] cascade unavailable -> main model only", flush=True)
        return None


def output_name(path: Path) -> str:
    """``<case>_0000.nii.gz`` in, ``<case>.nii.gz`` out."""
    stem = path.name
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith("_0000"):
        stem = stem[: -len("_0000")]
    return f"{stem}.nii.gz"


def find_cases(inputs: Path) -> list[Path]:
    found = sorted(p for p in inputs.glob("*.nii.gz") if not p.name.startswith("."))
    return found or sorted(p for p in inputs.glob("*.nii") if not p.name.startswith("."))


def run(inputs: Path, outputs: Path, model: Path = MODEL) -> int:
    """Segment every NIfTI in ``inputs`` and write uint8 masks to ``outputs``. Returns a process exit code."""
    outputs.mkdir(parents=True, exist_ok=True)
    todo = find_cases(inputs)
    print(f"[start] {len(todo)} case(s) in {inputs}", flush=True)
    if not todo:
        print(f"[warn] no NIfTI files found in {inputs}", flush=True)
        return 0
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    reader = SimpleITKIO()
    predictor = build_predictor(model)
    cascade = build_cascade()
    failures = 0
    started = time.time()
    for position, path in enumerate(todo, 1):
        destination = outputs / output_name(path)
        case_started = time.time()
        try:
            data, properties = reader.read_images([str(path)])
            target = np.array(predictor.configuration_manager.spacing, dtype=np.float64)
            mvox = float(np.prod(np.round(np.array(data.shape[1:]) * np.array(properties["spacing"]) / target))) / 1e6
            tta_this_case = USE_MIRRORING and mvox <= TTA_MAX_MVOX
            predictor.use_mirroring = tta_this_case
            _, probabilities = predictor.predict_single_npy_array(data, properties, None, None, True)
            predictor.use_mirroring = USE_MIRRORING
            foreground = np.asarray(probabilities[1], dtype=np.float32)
            voxel_ml = float(np.prod(properties["spacing"])) / 1000.0
            mask, emptied = postprocess(foreground, voxel_ml, BINARISE, PRUNE, EMPTY_TOTAL_ML, EMPTY_MAX_COMPONENTS)
            del probabilities
            lung_note = ""
            if cascade is not None:
                primary_s = time.time() - case_started
                if primary_s <= LUNG_TIME_GUARD_S:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        mask, info = cascade(np.asarray(data[0], dtype=np.float32), properties["spacing"], mask)
                        lung_note = (f" | lung {info['lung_ml']} mL"
                                     + (f" +{info['n_added']} comp/{info['added_voxels']} vox in {info['cascade_s']}s"
                                        if info["gated_in"] else " (gated out)"))
                    except Exception:
                        traceback.print_exc()
                        lung_note = " | lung cascade FAILED, main mask kept"
                else:
                    lung_note = f" | lung cascade skipped (main model {primary_s:.0f}s > guard)"
            reader.write_seg(mask, str(destination), properties)
            print(f"[{position}/{len(todo)}] {path.name} -> {destination.name} {int(mask.sum())} voxels"
                  f"{' (EMPTY RULE)' if emptied else ''} in {time.time() - case_started:.1f}s "
                  f"[{mvox:.0f} Mvox{'' if tta_this_case else ', TTA off'}]{lung_note}", flush=True)
        except Exception:
            traceback.print_exc()
            if GPU_RESAMPLE and torch.cuda.is_available():
                try:
                    set_resampling(predictor.configuration_manager, gpu=False)
                    data, properties = reader.read_images([str(path)])
                    _, probabilities = predictor.predict_single_npy_array(data, properties, None, None, True)
                    predictor.use_mirroring = USE_MIRRORING
                    mask, _ = postprocess(np.asarray(probabilities[1], dtype=np.float32),
                                          float(np.prod(properties["spacing"])) / 1000.0,
                                          BINARISE, PRUNE, EMPTY_TOTAL_ML, EMPTY_MAX_COMPONENTS)
                    reader.write_seg(mask, str(destination), properties)
                    print(f"[{position}/{len(todo)}] {path.name} -> {destination.name} {int(mask.sum())} voxels via "
                          f"CPU-resampling retry in {time.time() - case_started:.1f}s", flush=True)
                    set_resampling(predictor.configuration_manager, gpu=True, device=torch.device("cuda"))
                    continue
                except Exception:
                    traceback.print_exc()
                    set_resampling(predictor.configuration_manager, gpu=True, device=torch.device("cuda"))
            failures += 1
            try:
                data, properties = SimpleITKIO().read_images([str(path)])
                reader.write_seg(np.zeros(data.shape[1:], dtype=np.uint8), str(destination), properties)
                print(f"[{position}/{len(todo)}] {path.name} FAILED -> wrote empty mask", flush=True)
            except Exception:
                print(f"[{position}/{len(todo)}] {path.name} FAILED and no fallback could be written", flush=True)
    elapsed = time.time() - started
    print(f"[done] {len(todo)} case(s) in {elapsed:.1f}s ({elapsed / max(len(todo), 1):.2f}s/case), "
          f"{failures} failure(s)", flush=True)
    missing = [p for p in todo if not (outputs / output_name(p)).exists()]
    if missing:
        print(f"[error] {len(missing)} case(s) produced no output file", flush=True)
        return 1
    return 0
