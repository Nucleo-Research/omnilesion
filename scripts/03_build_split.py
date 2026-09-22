#!/usr/bin/env python3
"""Write the training split used by the trainer: a JSON file with ``train`` and ``sentinel`` case lists.

``train`` is every case of the raw dataset minus the evaluation cases you hold out; ``sentinel`` is the subset of
cases nnU-Net uses for its own patch-level validation during training (it is not used for model selection: the
checkpoint after the fixed budget is the model). The submitted model used data/split_train_sentinel.json:
16,975 training cases and a 200-case sentinel, with the 274 held-out cases of data/heldout_274.csv and a further
200 cases excluded from training.

    python scripts/03_build_split.py --dataset-id 501 --exclude data/heldout_274.csv --sentinel my_sentinel.txt \
        --out $nnUNet_preprocessed/Dataset501_OmniLesion/split_train_sentinel.json

Case-list files are CSV (column ``volume_id`` or ``case``) or one case id per line.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def read_cases(path: Path) -> set[str]:
    text = path.read_text().strip().splitlines()
    if not text:
        return set()
    if "," in text[0]:
        rows = list(csv.DictReader(text))
        column = "volume_id" if "volume_id" in rows[0] else "case"
        return {row[column].strip() for row in rows}
    return {line.strip() for line in text if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--nnunet-raw", type=Path, default=Path(os.environ.get("nnUNet_raw", "")))
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="case lists to keep out of training (evaluation sets); repeatable")
    parser.add_argument("--sentinel", type=Path, required=True,
                        help="case list used as nnU-Net's validation set; also excluded from training")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    dataset = next(arguments.nnunet_raw.glob(f"Dataset{arguments.dataset_id:03d}_*"))
    all_cases = {p.name[: -len(".nii.gz")] for p in (dataset / "labelsTr").glob("*.nii.gz")}
    excluded = set().union(*(read_cases(p) for p in arguments.exclude)) if arguments.exclude else set()
    sentinel = read_cases(arguments.sentinel) & all_cases
    train = sorted(all_cases - excluded - sentinel)
    assert not set(train) & sentinel
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps({"train": train, "sentinel": sorted(sentinel)}))
    print(f"train {len(train)}  sentinel {len(sentinel)}  excluded {len(excluded & all_cases)} -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
