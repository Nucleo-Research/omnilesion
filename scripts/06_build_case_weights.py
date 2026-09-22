#!/usr/bin/env python3
"""Turn the per-case organ attribution into the per-case sampling weight table read by the trainer.

Each case is credited to the family holding most of its lesion volume. With ``m_f`` the natural share of family ``f``
among the attributed cases, its target share is ``m_f ** (1 - alpha)`` (alpha 0 keeps the natural mixture, 1
equalises families); alpha 0.7 was used for the submission. Per-case weights are the family's target share divided
by its case count, rescaled so the lightest case has weight 1, and capped at ``--max-weight`` (40). Families that are
attribution artefacts rather than lesion sites (bone, muscle, vessel, heart, brain, spinal cord), families with fewer
than ``--min-pool-cases`` dominant cases, and cases without attribution keep their natural share; no family may fall
below ``--min-family-mass-ratio`` of its natural share. Frequencies come from the training labels only.

    python scripts/06_build_case_weights.py --attribution data/attribution.json --alpha 0.7 \
        --split $nnUNet_preprocessed/Dataset501_OmniLesion/split_train_sentinel.json --out case_weights_alpha07.json

``--split`` restricts the table to the training cases of the split (the trainer refuses tables naming other cases).
The table used for the submission is data/case_weights_alpha07.json. The printed natural-versus-resulting mass table
is the actual intervention; read it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max-weight", type=float, default=40.0)
    parser.add_argument("--min-family-mass-ratio", type=float, default=0.5)
    parser.add_argument("--min-pool-cases", type=int, default=30)
    parser.add_argument("--drop-families", default="bone,muscle,vessel,heart,brain,spinal_cord")
    parser.add_argument("--split", type=Path, default=None, help="split JSON; keep only its 'train' cases")
    args = parser.parse_args()

    payload = json.loads(args.attribution.read_text())
    per_case = payload["per_case"]
    dropped = {x for x in args.drop_families.split(",") if x}

    family_of_case: dict[str, str] = {}
    for case, record in per_case.items():
        families = record.get("families") or {}
        if record.get("empty") or not families:
            continue
        family_of_case[case] = max(families.items(), key=lambda kv: kv[1])[0]
    for case in per_case:
        family_of_case.setdefault(case, "no_attribution")

    counts = Counter(family_of_case.values())
    total = sum(counts.values())
    if not total:
        raise SystemExit("no attributed cases")
    natural = {family: count / total for family, count in counts.items()}
    target = {family: natural[family] ** (1.0 - args.alpha) for family in counts}

    protected = set(dropped) | {"no_attribution"} | {f for f, n in counts.items() if n < args.min_pool_cases}
    for family in protected & set(counts):
        target[family] = natural[family]
    scale = sum(target.values())
    target = {family: value / scale for family, value in target.items()}
    for _ in range(200):
        deficient = [f for f in counts if f not in protected and target[f] < args.min_family_mass_ratio * natural[f]]
        if not deficient:
            break
        for family in deficient:
            target[family] = args.min_family_mass_ratio * natural[family]
        free = [f for f in counts if f not in deficient and f not in protected]
        fixed_mass = sum(target[f] for f in deficient) + sum(target[f] for f in protected & set(counts))
        free_mass = sum(target[f] for f in free)
        if free_mass <= 0:
            break
        for family in free:
            target[family] = target[family] * (1.0 - fixed_mass) / free_mass

    per_case_weight = {family: target[family] / counts[family] for family in counts}
    smallest = min(per_case_weight.values())
    weight_of_family = {family: min(value / smallest, args.max_weight) for family, value in per_case_weight.items()}

    # A protected family keeps its natural share exactly when its weight equals the mean per-case weight.
    live = set(counts) - (protected & set(counts))
    total_cases = sum(counts.values())
    protected_cases = sum(counts[f] for f in protected & set(counts))
    if protected_cases and protected_cases < total_cases:
        free_mass = sum(weight_of_family[f] * counts[f] for f in live)
        mean_weight = (free_mass / (1.0 - protected_cases / total_cases)) / total_cases
        for family in protected & set(counts):
            weight_of_family[family] = min(mean_weight, args.max_weight)
    weights = {case: weight_of_family[family] for case, family in family_of_case.items()}

    if args.split is not None:
        train = set(json.loads(args.split.read_text())["train"])
        weights = {case: w for case, w in weights.items() if case in train}
        coverage = len(weights) / max(1, len(train))
        print(f"{len(weights)}/{len(train)} training cases covered by the attribution ({coverage:.1%})")

    mass_total = sum(weights.values())
    mass = defaultdict(float)
    for case, w in weights.items():
        mass[family_of_case[case]] += w / mass_total
    tiers = {family: {"cases": sorted(c for c, f in family_of_case.items() if f == family and c in weights),
                      "weight": round(weight_of_family.get(family, 1.0), 4)} for family in sorted(counts)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "purpose": "equalise organ-family exposure; alpha-tempered inverse frequency, training labels only",
        "alpha": args.alpha, "max_weight": args.max_weight, "min_family_mass_ratio": args.min_family_mass_ratio,
        "min_component_ml": payload["min_component_ml"], "min_containment": payload["min_containment"],
        "attributed_cases": len(family_of_case),
        "weights": {c: round(w, 4) for c, w in weights.items() if w > 1.0},
        "tiers": tiers,
    }, indent=1))

    print(f"alpha={args.alpha} max_weight={args.max_weight} floor={args.min_family_mass_ratio}; "
          f"{sum(1 for w in weights.values() if w > 1.0)} boosted cases\n")
    print(f"{'family':<14}{'cases':>7}{'weight':>9}{'natural':>10}{'resulting':>11}{'x':>7}")
    for family, count in counts.most_common():
        w = weight_of_family.get(family, 1.0)
        print(f"{family:<14}{count:7d}{w:9.2f}{natural[family]:10.2%}{mass[family]:11.2%}"
              f"{mass[family] / natural[family] if natural[family] else 0:7.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
