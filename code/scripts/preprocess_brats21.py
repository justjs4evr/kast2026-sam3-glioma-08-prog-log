#!/usr/bin/env python3
"""Preprocess BraTS2021 Task1 tar -> per-slice npz at native 240x240 (uint8).

Fixes vs student pipeline:
  - background stays exactly 0 (z-score on foreground, clip to [-5,5], map fg to [1,255])
  - val/test keep ALL brain-containing slices (enables 3D restack for HD95/NSD)
  - patient-level 70/15/15 split with fixed split seed (42), recorded to CSV
  - per-slice WT/TC/ET pixel counts recorded -> size tertiles computed on train only

Output layout ($OUT):
  slices/{patient_id}_{z:03d}.npz  keys: image uint8 [4,240,240] (t1,t1ce,t2,flair), mask uint8 [3,240,240]
  split_seed42.csv                 columns: filepath,patient_id,z,split,wt_px,tc_px,et_px,has_tumor
  stats.json                       train-set per-channel mean/std (on [0,1]-scaled uint8, brain only + full-frame both)
  tertiles.json                    P33.33/P66.67 per region over train tumor-positive slices
"""
import argparse, json, os, tarfile, glob, random
import numpy as np
import nibabel as nib
import pandas as pd
from multiprocessing import Pool
from functools import partial

MODS = ["t1", "t1ce", "t2", "flair"]


def norm_fg_uint8(vol: np.ndarray) -> np.ndarray:
    """Z-score on foreground, clip [-5,5], map fg to [1,255]; bg stays 0."""
    fg = vol > 0
    out = np.zeros_like(vol, dtype=np.uint8)
    if not fg.any():
        return out
    v = vol[fg].astype(np.float32)
    z = (v - v.mean()) / (v.std() + 1e-8)
    z = np.clip(z, -5.0, 5.0)
    scaled = ((z + 5.0) / 10.0) * 254.0 + 1.0  # [1,255]
    out[fg] = scaled.astype(np.uint8)
    return out


def process_patient(pdir: str, out_dir: str, split: str) -> list[dict]:
    pid = os.path.basename(pdir)
    try:
        vols = [norm_fg_uint8(nib.load(os.path.join(pdir, f"{pid}_{m}.nii.gz")).get_fdata()) for m in MODS]
        seg = nib.load(os.path.join(pdir, f"{pid}_seg.nii.gz")).get_fdata().astype(np.uint8)
    except FileNotFoundError:
        return []
    rows = []
    Z = seg.shape[2]
    for z in range(Z):
        s = seg[:, :, z]
        brain = any(v[:, :, z].any() for v in vols)
        if not brain:
            continue
        wt = np.isin(s, [1, 2, 4]).astype(np.uint8)
        tc = np.isin(s, [1, 4]).astype(np.uint8)
        et = (s == 4).astype(np.uint8)
        has_tumor = bool(wt.any())
        # train: tumor slices only; val/test: all brain slices (3D restack)
        if split == "train" and not has_tumor:
            continue
        img = np.stack([v[:, :, z] for v in vols], axis=0)          # [4,240,240] uint8
        msk = np.stack([wt, tc, et], axis=0)                        # [3,240,240] uint8
        fn = f"{pid}_{z:03d}.npz"
        np.savez_compressed(os.path.join(out_dir, "slices", fn), image=img, mask=msk)
        rows.append(dict(filepath=f"slices/{fn}", patient_id=pid, z=z, split=split,
                         wt_px=int(wt.sum()), tc_px=int(tc.sum()), et_px=int(et.sum()),
                         has_tumor=has_tumor))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing BraTS2021_* patient folders (extracted tar)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "slices"), exist_ok=True)
    patients = sorted([p for p in glob.glob(os.path.join(a.src, "BraTS2021_*")) if os.path.isdir(p)])
    print(f"patients: {len(patients)}")
    rng = random.Random(a.split_seed)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    split_of = {}
    for i, p in enumerate(shuffled):
        split_of[p] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    all_rows = []
    for split in ["train", "val", "test"]:
        plist = [p for p in patients if split_of[p] == split]
        with Pool(a.workers) as pool:
            res = pool.map(partial(process_patient, out_dir=a.out, split=split), plist)
        for r in res:
            all_rows.extend(r)
        print(f"{split}: {len(plist)} patients, {sum(len(r) for r in res)} slices")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(a.out, f"split_seed{a.split_seed}.csv"), index=False)

    # stats on train tumor slices (channel mean/std over full frame, [0,1] scale)
    tr = df[(df.split == "train")].sample(n=min(3000, (df.split == "train").sum()), random_state=0)
    s1 = np.zeros(4); s2 = np.zeros(4); cnt = 0
    for fp in tr.filepath:
        im = np.load(os.path.join(a.out, fp))["image"].astype(np.float64) / 255.0
        s1 += im.sum(axis=(1, 2)); s2 += (im ** 2).sum(axis=(1, 2)); cnt += im.shape[1] * im.shape[2]
    mean = (s1 / cnt); std = np.sqrt(s2 / cnt - mean ** 2)
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, open(os.path.join(a.out, "stats.json"), "w"), indent=1)

    # size tertiles (train, tumor-positive, per region) — pre-registered thresholds
    tert = {}
    trd = df[df.split == "train"]
    for reg, col in [("WT", "wt_px"), ("TC", "tc_px"), ("ET", "et_px")]:
        vals = trd[trd[col] > 0][col].values
        tert[reg] = {"P33.33": float(np.percentile(vals, 33.33)), "P66.67": float(np.percentile(vals, 66.67)),
                     "n_nonzero": int(len(vals))}
    json.dump(tert, open(os.path.join(a.out, "tertiles.json"), "w"), indent=1)
    print("stats:", mean.round(4).tolist(), std.round(4).tolist())
    print("tertiles:", tert)


if __name__ == "__main__":
    main()
