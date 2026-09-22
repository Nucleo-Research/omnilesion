#!/usr/bin/env python3
"""Segmentation-efficiency measurement in the challenge's format: seconds per case, peak GPU memory and the area under
the GPU memory-time curve, on cases spanning the size range of a folder.

Runs the same pipeline as the container natively (model initialised once and reported separately, since the harness
pays it once per container start). GPU memory is sampled with nvidia-smi every 0.25 s and attributed to this process by
subtracting the idle reading taken before the model is loaded. For harness-identical numbers run the container itself
once per case and sample nvidia-smi from the host.

    python scripts/evaluation/efficiency_bench.py --images /data/validation_images --model <model folder> \
        --lung-dir <lung weights> --n-cases 8 --out bench/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class MemorySampler(threading.Thread):
    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self.stop_flag = threading.Event()

    @staticmethod
    def read() -> float:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True)
        return float(out.stdout.strip().splitlines()[0])

    def run(self) -> None:
        while not self.stop_flag.is_set():
            self.samples.append((time.time(), self.read()))
            time.sleep(self.interval)

    def window(self, start: float, end: float, baseline: float) -> tuple[float, float]:
        pts = [(t, m - baseline) for t, m in self.samples if start <= t <= end]
        if len(pts) < 2:
            return (max((m for _, m in pts), default=0.0), 0.0)
        peak = max(m for _, m in pts)
        area = sum((pts[i + 1][0] - pts[i][0]) * (pts[i][1] + pts[i + 1][1]) / 2 for i in range(len(pts) - 1))
        return peak, area


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--lung-dir", type=Path, default=None)
    parser.add_argument("--n-cases", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    os.environ["OMNILESION_MODEL"] = str(arguments.model)
    os.environ["OMNILESION_FOLD"] = arguments.fold
    if arguments.lung_dir is None:
        os.environ["OMNILESION_LUNG_CASCADE"] = "0"
    else:
        os.environ["OMNILESION_LUNG_DIR"] = str(arguments.lung_dir)
    import nibabel as nib
    import torch
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    from omnilesion import inference
    from omnilesion.postprocess import postprocess

    images = sorted(arguments.images.glob("*_0000.nii.gz")) or sorted(arguments.images.glob("*.nii.gz"))
    sizes = sorted((int(np.prod(nib.load(str(p)).shape)), p) for p in images)
    picks = [sizes[int(round(i * (len(sizes) - 1) / max(arguments.n_cases - 1, 1)))][1] for i in range(arguments.n_cases)]
    print("cases:", [p.name for p in picks], flush=True)

    sampler = MemorySampler()
    sampler.start()
    time.sleep(1.5)
    baseline = float(np.median([m for _, m in sampler.samples])) if sampler.samples else sampler.read()
    reader = SimpleITKIO()
    t0 = time.time()
    predictor = inference.build_predictor(arguments.model)
    cascade = inference.build_cascade()
    init_seconds = time.time() - t0
    time.sleep(1.0)
    init_peak, _ = sampler.window(t0, time.time(), baseline)
    print(f"initialisation {init_seconds:.1f} s, resident {init_peak:.0f} MB above idle {baseline:.0f} MB", flush=True)

    rows = []
    for path in picks:
        nifti = nib.load(str(path))
        shape = tuple(int(s) for s in nifti.shape)
        start = time.time()
        data, properties = reader.read_images([str(path)])
        _, probabilities = predictor.predict_single_npy_array(data, properties, None, None, True)
        mask, _ = postprocess(np.asarray(probabilities[1], dtype=np.float32),
                              float(np.prod(properties["spacing"])) / 1000.0)
        if cascade is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask, _ = cascade(np.asarray(data[0], dtype=np.float32), properties["spacing"], mask)
        reader.write_seg(mask, str(arguments.out / inference.output_name(path)), properties)
        end = time.time()
        time.sleep(0.6)
        peak, area = sampler.window(start, end, baseline)
        row = {"case": path.name.replace("_0000.nii.gz", ""), "shape": "x".join(map(str, shape)),
               "voxels": int(np.prod(shape)), "seconds": round(end - start, 2), "max_gpu_mb": round(peak),
               "total_gpu_mb_s": round(area)}
        rows.append(row)
        print(row, flush=True)
    sampler.stop_flag.set()
    summary = {"device": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.strip(),
               "init_seconds": round(init_seconds, 1), "init_resident_mb": round(init_peak),
               "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 2),
               "max_seconds": round(max(r["seconds"] for r in rows), 2),
               "max_gpu_mb": round(max(r["max_gpu_mb"] for r in rows)),
               "mean_total_gpu_mb_s": round(float(np.mean([r["total_gpu_mb_s"] for r in rows]))), "cases": rows}
    (arguments.out / "efficiency.json").write_text(json.dumps(summary, indent=2))
    with (arguments.out / "efficiency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
