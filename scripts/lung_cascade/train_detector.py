#!/usr/bin/env python3
"""Train the 3D RetinaNet lung-nodule detector from scratch on the cache of prepare_nodule_data.py.

Architecture: omnilesion.lung_cascade.detector (MONAI RetinaNet, 3D ResNet-50, FPN, cubic 4/6/8 mm anchors, ATSS
matcher, hard-negative sampler). Sampler: 128^3 crops at 1 mm, half of them centred (with jitter) on a random
reference box, half uniform; flips on all axes; light intensity jitter. SGD (momentum 0.9, weight decay 3e-5,
Nesterov), 1000-iteration warm-up then cosine decay, mixed precision, gradient clipping at 10.
Validation every ``--val-every`` iterations: full-volume sliding-window detection on 3 % (at least 20) of the cases
held back from the cache, reporting recall at several score cuts and false positives per scan.

The submission trained for 300,000 iterations at batch 4 (~7 h on one H100) and used ``ckpt_300000.pt``:

    python scripts/lung_cascade/train_detector.py --cache /fast_disk/nodule_cache --out /fast_disk/detector \
        --iters 300000 --batch 4 --val-every 20000
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omnilesion.lung_cascade.detector import build_detector  # noqa: E402


class CropDataset(Dataset):
    """Infinite sampler of (image crop, boxes) from the uint8 cache."""

    def __init__(self, cache: Path, cases: list[str], patch, n_per_epoch: int, pos_frac=0.5):
        self.cache, self.cases, self.patch, self.n, self.pos_frac = cache, cases, np.array(patch), n_per_epoch, pos_frac
        self.meta = {c: json.load((cache / "det" / f"{c}.json").open()) for c in cases}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random((i * 7919 + torch.initial_seed()) % (2**31))
        c = rng.choice(self.cases)
        m = self.meta[c]
        vol = np.load(self.cache / "det" / f"{c}.npy", mmap_mode="r")
        shape = np.array(vol.shape)
        boxes = np.array(m["boxes_xyzxyz"], dtype=np.float32).reshape(-1, 6)
        P = self.patch
        if len(boxes) and rng.random() < self.pos_frac:
            b = boxes[rng.randrange(len(boxes))]
            cen = (b[:3] + b[3:]) / 2
            jit = np.array([rng.uniform(-0.35, 0.35) for _ in range(3)]) * P
            lo = np.round(cen + jit - P / 2).astype(int)
        else:
            lo = np.array([rng.randint(0, max(int(s - p), 0)) for s, p in zip(shape, P)])
        lo = np.clip(lo, 0, np.maximum(shape - P, 0))
        hi = lo + P
        crop = np.zeros(P, dtype=np.uint8)
        src = vol[lo[0]:min(hi[0], shape[0]), lo[1]:min(hi[1], shape[1]), lo[2]:min(hi[2], shape[2])]
        crop[: src.shape[0], : src.shape[1], : src.shape[2]] = src
        if len(boxes):
            bb = boxes.copy()
            bb[:, :3] -= lo
            bb[:, 3:] -= lo
            cen = (bb[:, :3] + bb[:, 3:]) / 2
            keep = np.all(cen >= 0, 1) & np.all(cen < P, 1)
            bb = bb[keep]
            bb[:, :3] = np.clip(bb[:, :3], 0, P)
            bb[:, 3:] = np.clip(bb[:, 3:], 0, P)
            bb = bb[np.all(bb[:, 3:] - bb[:, :3] >= 1.0, 1)]
        else:
            bb = np.zeros((0, 6), dtype=np.float32)
        img = crop.astype(np.float32) / 255.0
        for ax in range(3):
            if rng.random() < 0.5:
                img = np.flip(img, axis=ax)
                if len(bb):
                    lo_ax = P[ax] - bb[:, ax + 3]
                    hi_ax = P[ax] - bb[:, ax]
                    bb[:, ax] = lo_ax
                    bb[:, ax + 3] = hi_ax
        if rng.random() < 0.15:
            img = np.clip(img * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img))[None]
        return img, torch.from_numpy(np.ascontiguousarray(bb)).float(), torch.zeros(len(bb), dtype=torch.long)


def collate(batch):
    return [b[0] for b in batch], [{"box": b[1], "label": b[2]} for b in batch]


@torch.no_grad()
def validate(det, cache: Path, cases: list[str], device, cuts=(0.3, 0.5, 0.7, 0.8, 0.9, 0.95), max_cases=40):
    det.eval()
    stats = {c: {"tp": 0, "fp": 0, "n": 0} for c in cuts}
    n_gt = 0
    t0 = time.time()
    for c in cases[:max_cases]:
        m = json.load((cache / "det" / f"{c}.json").open())
        vol = np.load(cache / "det" / f"{c}.npy").astype(np.float32) / 255.0
        img = torch.from_numpy(vol)[None]
        with torch.autocast("cuda"):
            out = det([img.to(device)], use_inferer=True)[0]
        boxes = out["box"].float().cpu().numpy().reshape(-1, 6)
        scores = out["label_scores"].float().cpu().numpy().ravel()
        gts = [(np.array(x["centroid_crop_mm"]), (3 * x["vol_ml"] * 1000 / (4 * math.pi)) ** (1 / 3))
               for x in m["components"] if x["in_crop"]]
        n_gt += len(gts)
        for cut in cuts:
            sel = boxes[scores >= cut]
            cen = (sel[:, :3] + sel[:, 3:]) / 2 - 0.5 if len(sel) else np.zeros((0, 3))
            hit = np.zeros(len(gts), bool)
            fp = 0
            for p in cen:
                d = [np.linalg.norm(p - g[0]) for g in gts]
                j = int(np.argmin(d)) if d else -1
                if j >= 0 and d[j] <= gts[j][1] + 5.0:
                    hit[j] = True
                else:
                    fp += 1
            stats[cut]["tp"] += int(hit.sum())
            stats[cut]["fp"] += fp
            stats[cut]["n"] += 1
    det.train()
    res = {f"{cut:.2f}": {"recall": stats[cut]["tp"] / max(n_gt, 1), "fp_per_scan": stats[cut]["fp"] / max(stats[cut]["n"], 1)}
           for cut in cuts}
    res.update({"n_gt": n_gt, "n_cases": min(len(cases), max_cases), "sec": round(time.time() - t0, 1)})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=300000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=20000)
    ap.add_argument("--val-cases", type=int, default=40)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    cache, out = a.cache, a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    rows = [json.loads(l) for l in (cache / "manifest.jsonl").open()]
    ok = sorted(r["case"] for r in rows if r["status"] == "ok" and r["n_boxes"] > 0)
    rng = random.Random(0)
    rng.shuffle(ok)
    n_val = max(20, int(0.03 * len(ok)))
    val_cases, train_cases = ok[:n_val], ok[n_val:]
    json.dump({"train": sorted(train_cases), "val": sorted(val_cases)}, (out / "split.json").open("w"))
    print(f"train {len(train_cases)} cases, val {len(val_cases)}", flush=True)
    det = build_detector(device, a.patch, training=True, sw_batch_size=1)
    ds = CropDataset(cache, train_cases, a.patch, n_per_epoch=a.iters * a.batch)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate, pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    opt = torch.optim.SGD(det.network.parameters(), lr=a.lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(it, a.iters) / a.iters)))
    scaler = torch.amp.GradScaler("cuda")
    it0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        det.network.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        it0 = ck["iter"]
        for _ in range(it0):
            sched.step()
    det.train()
    t0 = time.time()
    run = {"cls": 0.0, "reg": 0.0, "n": 0}
    log = (out / "train_log.jsonl").open("a")
    for it, (imgs, targets) in enumerate(dl, start=it0):
        if it >= a.iters:
            break
        imgs = [x.to(device, non_blocking=True) for x in imgs]
        targets = [{"box": t["box"].to(device), "label": t["label"].to(device)} for t in targets]
        with torch.autocast("cuda"):
            losses = det(imgs, targets)
            loss = losses["classification"] + losses["box_regression"]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(det.network.parameters(), 10.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run["cls"] += float(losses["classification"])
        run["reg"] += float(losses["box_regression"])
        run["n"] += 1
        if (it + 1) % 100 == 0:
            rec = {"iter": it + 1, "cls": run["cls"] / run["n"], "reg": run["reg"] / run["n"], "lr": sched.get_last_lr()[0],
                   "it_per_s": run["n"] / (time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run = {"cls": 0.0, "reg": 0.0, "n": 0}
            t0 = time.time()
        if (it + 1) % a.val_every == 0 or it + 1 == a.iters:
            torch.save({"model": det.network.state_dict(), "opt": opt.state_dict(), "iter": it + 1, "patch": a.patch},
                       out / "last.pt")
            v = validate(det, cache, val_cases, device, max_cases=a.val_cases)
            v["iter"] = it + 1
            log.write(json.dumps({"val": v}) + "\n")
            log.flush()
            print("VAL", json.dumps(v), flush=True)
            torch.save({"model": det.network.state_dict(), "iter": it + 1, "patch": a.patch}, out / f"ckpt_{it + 1:06d}.pt")
            t0 = time.time()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
