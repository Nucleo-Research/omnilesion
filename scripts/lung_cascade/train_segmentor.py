#!/usr/bin/env python3
"""Train the 65^3 lung-nodule patch segmentor (omnilesion.lung_cascade.nodule_unet.UNet3D, 12 base channels).

Data: the 81^3 patches of prepare_nodule_data.py. Each training sample is a 65^3 crop whose centre is jittered by up
to 5 mm (8 mm for 20 % of the samples), so the network learns to segment the nodule near the centre, which is what a
detector proposal gives it; flips on all axes; light intensity jitter and noise. Loss: soft Dice + BCE. AdamW
(lr 1e-3, weight decay 1e-4), 500-iteration warm-up then cosine decay, mixed precision. Validation every
``--val-every`` iterations on 3 % (at least 20) of the cases: Dice with the inference post-processing (threshold,
centre component) at jitter 0 and 3 mm; the best checkpoint by the mean of the two is kept with its manifest.

The submission trained for 40,000 iterations at batch 16 (~40 min on one H100):

    python scripts/lung_cascade/train_segmentor.py --cache /fast_disk/nodule_cache --out /fast_disk/segmentor
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omnilesion.lung_cascade.nodule_unet import UNet3D, center_component, normalize_hu  # noqa: E402

P_IN, P_OUT = 81, 65


class PatchDataset(Dataset):
    def __init__(self, files, n, jitter=5.0, big_jitter=8.0, big_frac=0.2, train=True):
        self.files, self.n, self.j, self.bj, self.bf, self.train = files, n, jitter, big_jitter, big_frac, train

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random((i * 7919 + torch.initial_seed()) % (2**31))
        z = np.load(self.files[i % len(self.files)] if not self.train else rng.choice(self.files))
        hu, m = z["hu"].astype(np.float32), z["mask"].astype(np.float32)
        j = self.bj if rng.random() < self.bf else self.j
        off = np.array([rng.uniform(-j, j) for _ in range(3)]) if self.train else np.zeros(3)
        lo = np.clip(np.round((P_IN - P_OUT) / 2 + off).astype(int), 0, P_IN - P_OUT)
        hu = hu[lo[0]:lo[0] + P_OUT, lo[1]:lo[1] + P_OUT, lo[2]:lo[2] + P_OUT]
        m = m[lo[0]:lo[0] + P_OUT, lo[1]:lo[1] + P_OUT, lo[2]:lo[2] + P_OUT]
        if self.train:
            for ax in range(3):
                if rng.random() < 0.5:
                    hu, m = np.flip(hu, ax), np.flip(m, ax)
            if rng.random() < 0.2:
                hu = hu * rng.uniform(0.9, 1.1) + rng.uniform(-40, 40)
            if rng.random() < 0.1:
                hu = hu + np.random.default_rng(rng.randrange(2**31)).normal(0, 15, hu.shape)
        x = normalize_hu(np.ascontiguousarray(hu))
        return torch.from_numpy(x)[None], torch.from_numpy(np.ascontiguousarray(m))[None], float(z["vol_ml"])


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3, 4)) + eps
    den = p.sum((1, 2, 3, 4)) + target.sum((1, 2, 3, 4)) + eps
    return (1 - num / den).mean()


@torch.no_grad()
def validate(model, files, device, threshold, jitter_mm, max_n=400):
    model.eval()
    rng = np.random.default_rng(0)
    dices, vols = [], []
    for f in files[:max_n]:
        z = np.load(f)
        hu, m = z["hu"].astype(np.float32), z["mask"].astype(bool)
        off = rng.uniform(-jitter_mm, jitter_mm, 3) if jitter_mm else np.zeros(3)
        lo = np.clip(np.round((P_IN - P_OUT) / 2 + off).astype(int), 0, P_IN - P_OUT)
        hu = hu[lo[0]:lo[0] + P_OUT, lo[1]:lo[1] + P_OUT, lo[2]:lo[2] + P_OUT]
        m = m[lo[0]:lo[0] + P_OUT, lo[1]:lo[1] + P_OUT, lo[2]:lo[2] + P_OUT]
        x = torch.from_numpy(normalize_hu(np.ascontiguousarray(hu)))[None, None].to(device)
        with torch.autocast("cuda"):
            pr = torch.sigmoid(model(x))[0, 0].float().cpu().numpy()
        pm = center_component(pr >= threshold)
        dices.append(2 * (pm & m).sum() / max(pm.sum() + m.sum(), 1))
        vols.append(float(z["vol_ml"]))
    model.train()
    d, v = np.array(dices), np.array(vols)
    band = lambda sel: float(d[sel].mean()) if sel.any() else None  # noqa: E731
    return {"dice_mean": float(d.mean()), "dice_median": float(np.median(d)), "zero": int((d == 0).sum()), "n": len(d),
            "dice_lt0.1ml": band(v < 0.1), "dice_0.1-1ml": band((v >= 0.1) & (v < 1)), "dice_ge1ml": band(v >= 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--base-channels", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    files = sorted((a.cache / "seg").glob("*.npz"))
    cases = sorted({f.name.rsplit("_", 1)[0] for f in files})
    rng = random.Random(0)
    rng.shuffle(cases)
    val_cases = set(cases[: max(20, int(0.03 * len(cases)))])
    val_files = [f for f in files if f.name.rsplit("_", 1)[0] in val_cases]
    train_files = [f for f in files if f.name.rsplit("_", 1)[0] not in val_cases]
    json.dump({"val_cases": sorted(val_cases), "n_train_patches": len(train_files), "n_val_patches": len(val_files)},
              (out / "split.json").open("w"))
    print(f"train patches {len(train_files)} ({len(cases) - len(val_cases)} cases), val patches {len(val_files)}", flush=True)
    model = UNet3D(in_channels=1, base_channels=a.base_channels).to(device)
    model.train()
    dl = DataLoader(PatchDataset(train_files, a.iters * a.batch), batch_size=a.batch, num_workers=a.workers,
                    pin_memory=True, persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    scaler = torch.amp.GradScaler("cuda")
    log = (out / "train_log.jsonl").open("a")
    t0 = time.time()
    run = [0.0, 0]
    best = -1
    for it, (x, y, _) in enumerate(dl):
        if it >= a.iters:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda"):
            logits = model(x)
            loss = dice_loss(logits.float(), y) + F.binary_cross_entropy_with_logits(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run[0] += float(loss)
        run[1] += 1
        if (it + 1) % 200 == 0:
            rec = {"iter": it + 1, "loss": run[0] / run[1], "lr": sched.get_last_lr()[0], "it_per_s": run[1] / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run = [0.0, 0]
            t0 = time.time()
        if (it + 1) % a.val_every == 0 or it + 1 == a.iters:
            v0 = validate(model, val_files, device, a.threshold, 0.0)
            v3 = validate(model, val_files, device, a.threshold, 3.0)
            log.write(json.dumps({"val": {"iter": it + 1, "jitter0": v0, "jitter3": v3}}) + "\n")
            log.flush()
            print("VAL", json.dumps({"iter": it + 1, "jitter0": v0, "jitter3": v3}), flush=True)
            torch.save(model.state_dict(), out / "last_state_dict.pt")
            score = (v0["dice_mean"] + v3["dice_mean"]) / 2
            if score > best:
                best = score
                torch.save(model.state_dict(), out / "segmentor_state_dict.pt")
                json.dump({"model_name": "omnilesion-nodule-unet", "version": time.strftime("v%Y-%m-%d"),
                           "task": "lung_nodule_segmentation",
                           "model": {"architecture": "UNet3D", "base_channels": a.base_channels, "in_channels": 1,
                                     "input_spacing_mm": [1.0, 1.0, 1.0],
                                     "normalization": {"clip_hu": [-1000.0, 600.0], "scale": "((clip + 1000) / 1600) * 2 - 1"},
                                     "patch_size": [65, 65, 65]},
                           "postprocessing": {"keep_center_component": True, "threshold": a.threshold},
                           "training": {"data": "FLARE 2026 Task 1 nodule cohorts (training split): LUNA25, LIDC-IDRI, LNDb",
                                        "iters": it + 1, "batch": a.batch, "jitter_mm": [5.0, 8.0]},
                           "selection": {"iter": it + 1, "val_dice_j0": v0["dice_mean"], "val_dice_j3": v3["dice_mean"]}},
                          (out / "segmentor_manifest.json").open("w"), indent=1)
            t0 = time.time()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
