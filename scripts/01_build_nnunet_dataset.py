#!/usr/bin/env python3
"""Build the nnU-Net raw dataset from the FLARE 2026 Task 1 labelled training release.

The release ships images and labels in two parts::

    <flare_root>/train_label/imagesTr_part1/<case>_0000.nii.gz   <flare_root>/train_label/labelsTr_part1/<case>.nii.gz
    <flare_root>/train_label/imagesTr_part2/...                    <flare_root>/train_label/labelsTr_part2/...

This script creates ``$nnUNet_raw/Dataset501_OmniLesion/{imagesTr,labelsTr}`` as a symlink view over both parts
(nothing is copied), writes ``dataset.json`` (one CT channel, binary lesion label), and repairs the few files whose
NIfTI direction cosines are not exactly orthonormal, which SimpleITK rejects: for those cases only, a fixed copy is
written under ``<dataset>/header_fixed/`` with the direction matrix replaced by its nearest orthonormal matrix, the voxel
data untouched and the label cast to uint8, and the symlink points to the copy. Labels are used exactly as released:
partially annotated cases are not masked.

    python scripts/01_build_nnunet_dataset.py --flare-root /data/FLARE2026-Task1 --dataset-id 501
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

DATASET_NAME = "Dataset{id:03d}_OmniLesion"


def nearest_orthonormal(direction: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(direction)
    return u @ vt


def header_is_orthonormal(path: Path, tol: float = 1e-6) -> bool:
    """SimpleITK's own criterion: the direction matrix (affine columns divided by spacing) must be orthonormal."""
    img = nib.load(str(path))
    affine = img.affine[:3, :3]
    spacing = np.linalg.norm(affine, axis=0)
    direction = affine / spacing
    return bool(np.allclose(direction.T @ direction, np.eye(3), atol=tol))


def fix_header(source: Path, destination: Path, is_label: bool) -> None:
    img = nib.load(str(source))
    affine = img.affine.copy()
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    affine[:3, :3] = nearest_orthonormal(affine[:3, :3] / spacing) * spacing
    data = np.asarray(img.dataobj)
    if is_label:
        data = np.rint(data).astype(np.uint8)
    fixed = nib.Nifti1Image(data, affine)
    fixed.header.set_zooms(spacing)
    nib.save(fixed, str(destination))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flare-root", type=Path, required=True, help="folder containing train_label/")
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--nnunet-raw", type=Path, default=Path(os.environ.get("nnUNet_raw", "")),
                        help="defaults to $nnUNet_raw")
    arguments = parser.parse_args()
    if not str(arguments.nnunet_raw):
        raise SystemExit("set nnUNet_raw or pass --nnunet-raw")

    root = arguments.flare_root / "train_label"
    image_dirs = sorted(root.glob("imagesTr_part*")) or [root / "imagesTr"]
    label_dirs = sorted(root.glob("labelsTr_part*")) or [root / "labelsTr"]
    images = {p.name[: -len("_0000.nii.gz")]: p for d in image_dirs for p in d.glob("*_0000.nii.gz")}
    labels = {p.name[: -len(".nii.gz")]: p for d in label_dirs for p in d.glob("*.nii.gz")}
    cases = sorted(set(images) & set(labels))
    print(f"{len(images)} images, {len(labels)} labels, {len(cases)} paired cases")

    dataset = arguments.nnunet_raw / DATASET_NAME.format(id=arguments.dataset_id)
    (dataset / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset / "labelsTr").mkdir(parents=True, exist_ok=True)
    fixed_dir = dataset / "header_fixed"
    fixed = []
    for case in cases:
        image, label = images[case], labels[case]
        if not (header_is_orthonormal(image) and header_is_orthonormal(label)):
            (fixed_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
            (fixed_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
            fix_header(image, fixed_dir / "imagesTr" / image.name, is_label=False)
            fix_header(label, fixed_dir / "labelsTr" / label.name, is_label=True)
            image, label = fixed_dir / "imagesTr" / image.name, fixed_dir / "labelsTr" / label.name
            fixed.append(case)
        for source, target in ((image, dataset / "imagesTr" / f"{case}_0000.nii.gz"),
                               (label, dataset / "labelsTr" / f"{case}.nii.gz")):
            if target.is_symlink() or target.exists():
                target.unlink()
            target.symlink_to(source.resolve())

    (dataset / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "dataset_name": dataset.name,
        "description": "Symlink view of the FLARE 2026 Task 1 labelled training set (binary lesion label).",
    }, indent=4) + "\n")
    (dataset / "header_fix_manifest.json").write_text(json.dumps(fixed, indent=1) + "\n")
    print(f"wrote {dataset}; {len(fixed)} case(s) with repaired headers: {fixed[:5]}{'...' if len(fixed) > 5 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
