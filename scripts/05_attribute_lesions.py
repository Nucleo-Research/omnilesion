#!/usr/bin/env python3
"""Attribute every reference lesion component of the training set to an organ family.

Attribution is by containment: a 26-connected component of the label is credited to the TotalSegmentator organ that
holds the largest share of its voxels, provided that share is at least ``--min-containment`` (0.5); otherwise it is
``unassigned``. Components below ``--min-component-ml`` (0.5 mL) are ignored. Organs are collapsed into families
(liver, lung, kidney, pancreas, colon, ...). The output feeds scripts/06_build_case_weights.py.

    python scripts/05_attribute_lesions.py --labels $nnUNet_raw/Dataset501_OmniLesion/labelsTr \
        --totalseg /data/totalseg --out data/attribution.json

The submitted model's attribution was computed on the 3 x 2 x 2 mm training grid; this script works on the native
label grid, which gives the same families with slightly different millilitre counts. The table actually used is
shipped as data/attribution.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

FAMILY_OF = {
    "colon": "colon", "small_bowel": "small_bowel", "duodenum": "small_bowel",
    "stomach": "stomach", "esophagus": "esophagus", "urinary_bladder": "bladder",
    "prostate": "prostate", "liver": "liver", "gallbladder": "biliary", "pancreas": "pancreas",
    "spleen": "spleen", "kidney_left": "kidney", "kidney_right": "kidney",
    "kidney_cyst_left": "kidney", "kidney_cyst_right": "kidney",
    "adrenal_gland_left": "adrenal", "adrenal_gland_right": "adrenal",
    "thyroid_gland": "thyroid", "trachea": "airway", "brain": "brain", "spinal_cord": "spinal_cord",
}
PREFIX_FAMILY = (
    (("lung",), "lung"),
    (("vertebrae", "rib", "sacrum", "hip", "femur", "humerus", "scapula", "clavicula", "skull",
      "sternum", "costal_cartilages"), "bone"),
    (("iliopsoas", "gluteus", "autochthon"), "muscle"),
    (("aorta", "iliac", "vena", "portal", "pulmonary", "brachiocephalic", "subclavian",
      "common_carotid", "atrial", "superior_vena", "inferior_vena"), "vessel"),
    (("heart", "myocardium", "ventricle", "atrium"), "heart"),
)


def family_of(name: str) -> str:
    if name in FAMILY_OF:
        return FAMILY_OF[name]
    for prefixes, family in PREFIX_FAMILY:
        if name.startswith(prefixes):
            return family
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="labelsTr folder")
    parser.add_argument("--totalseg", type=Path, required=True, help="TotalSegmentator multilabel outputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-containment", type=float, default=0.50)
    parser.add_argument("--min-component-ml", type=float, default=0.5)
    arguments = parser.parse_args()

    from totalsegmentator.map_to_binary import class_map

    names = class_map["total"]
    per_case: dict[str, dict] = {}
    family_cases: dict[str, set[str]] = defaultdict(set)
    family_components: Counter = Counter()
    family_ml: dict[str, float] = defaultdict(float)
    skipped: Counter = Counter()

    paths = sorted(arguments.labels.glob("*.nii.gz"))
    for position, label_path in enumerate(paths, 1):
        case = label_path.name[: -len(".nii.gz")]
        organ_path = arguments.totalseg / f"{case}.nii.gz"
        if not organ_path.exists():
            skipped["no_totalseg"] += 1
            continue
        label_img = nib.load(str(label_path))
        seg = np.asarray(label_img.dataobj) > 0
        organs = np.asarray(nib.load(str(organ_path)).dataobj).astype(np.int16)
        if seg.shape != organs.shape:
            skipped["shape_mismatch"] += 1
            continue
        voxel_ml = float(np.prod(label_img.header.get_zooms()[:3])) / 1000.0
        if not seg.any():
            per_case[case] = {"families": {}, "empty": True}
            skipped["empty_label"] += 1
            continue
        labelled, count = ndimage.label(seg, structure=np.ones((3, 3, 3)))
        found: dict[str, float] = defaultdict(float)
        for index in range(1, count + 1):
            component = labelled == index
            size = int(component.sum())
            millilitres = size * voxel_ml
            if millilitres < arguments.min_component_ml:
                continue
            inside = organs[component]
            present, counts = np.unique(inside[inside > 0], return_counts=True)
            if len(present) and counts.max() / size >= arguments.min_containment:
                family = family_of(names.get(int(present[np.argmax(counts)]), "other"))
            else:
                family = "unassigned"
            found[family] += millilitres
            family_components[family] += 1
        per_case[case] = {"families": {k: round(v, 2) for k, v in found.items()}, "empty": False}
        for family, millilitres in found.items():
            family_cases[family].add(case)
            family_ml[family] += millilitres
        if position % 1000 == 0:
            print(f"  {position}/{len(paths)}", flush=True)

    families = sorted(family_cases, key=lambda f: -len(family_cases[f]))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({
        "min_containment": arguments.min_containment,
        "min_component_ml": arguments.min_component_ml,
        "cases_attributed": len(per_case),
        "skipped": dict(skipped),
        "summary": {f: {"cases": len(family_cases[f]), "components": family_components[f],
                        "total_ml": round(family_ml[f], 1)} for f in families},
        "per_case": per_case,
    }, indent=1))
    total = sum(len(family_cases[f]) for f in families) or 1
    print(f"attributed {len(per_case)} cases; skipped {dict(skipped)}")
    print(f"{'family':<14}{'cases':>8}{'share':>9}{'comps':>8}{'total mL':>11}")
    for family in families:
        n = len(family_cases[family])
        print(f"{family:<14}{n:8d}{n / total:9.2%}{family_components[family]:8d}{family_ml[family]:11.0f}")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
