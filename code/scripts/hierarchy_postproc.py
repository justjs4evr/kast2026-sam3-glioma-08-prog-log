#!/usr/bin/env python3
"""Hierarchy-enforcement postprocessing ablation on SAVED prediction masks (no GPU).

Modes:
  none : as saved
  hard : ET <- ET∩TC∩WT ; TC <- TC∩WT  (binary intersection)
  soft : child prob <- min(child, parent) then threshold 0.5 (monotone prob calibration)

Reads <run>/preds_<split>_<family>/*.npz (pred uint8 [3,H,W], prob uint8 0-255),
restacks per patient, computes 3D dice/HD95/NSD/violations per mode.
Output: <run>/postproc_<split>_<family>.parquet + summary json.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/workspace/kast")
from kast2.data import load_split
from kast2.metrics import REGIONS, dice_per_channel, hierarchy_violation, surface_metrics_3d


def enforce(prob: np.ndarray, mode: str) -> np.ndarray:
    """prob: [3,H,W] float in [0,1] (WT,TC,ET). Returns binary [3,H,W]."""
    if mode == "none":
        return (prob >= 0.5).astype(np.float32)
    if mode == "soft":
        wt = prob[0]
        tc = np.minimum(prob[1], wt)
        et = np.minimum(prob[2], tc)
        return (np.stack([wt, tc, et]) >= 0.5).astype(np.float32)
    if mode == "hard":
        b = (prob >= 0.5)
        wt = b[0]
        tc = b[1] & wt
        et = b[2] & tc
        return np.stack([wt, tc, et]).astype(np.float32)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data", default="/workspace/data/brats21")
    ap.add_argument("--split", default="val")
    ap.add_argument("--family", default="short")
    a = ap.parse_args()

    mask_dir = os.path.join(a.run_dir, f"preds_{a.split}_{a.family}")
    files = sorted(glob.glob(os.path.join(mask_dir, "*.npz")))
    assert files, f"no masks in {mask_dir}"

    # GT from dataset npz (native 240)
    df = load_split(a.data, a.split).set_index(["patient_id", "z"])

    probs_by_pat = defaultdict(dict)
    for f in files:
        base = os.path.basename(f)[:-4]
        pid, z = base.rsplit("_", 1)
        probs_by_pat[pid][int(z)] = np.load(f)["prob"].astype(np.float32) / 255.0

    rows = []
    for pid, zmap in probs_by_pat.items():
        zs = sorted(zmap)
        gt = []
        for z in zs:
            fp = df.loc[(pid, z)].filepath
            gt.append(np.load(os.path.join(a.data, fp))["mask"].astype(np.float32))
        gv = torch.from_numpy(np.stack(gt, axis=1))  # [3,D,H,W]
        for mode in ("none", "hard", "soft"):
            pv = torch.from_numpy(np.stack([enforce(zmap[z], mode) for z in zs], axis=1))
            r = {"patient_id": pid, "mode": mode}
            d3 = dice_per_channel(pv, gv).numpy()
            r.update({f"dice3d_{x}": d3[i] for i, x in enumerate(REGIONS)})
            r.update(surface_metrics_3d(pv, gv))
            r.update(hierarchy_violation(pv))
            rows.append(r)

    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(a.run_dir, f"postproc_{a.split}_{a.family}.parquet"))
    summ = {}
    for mode in ("none", "hard", "soft"):
        sub = out[out["mode"] == mode]
        s = {}
        for x in REGIONS:
            s[f"dice3d_{x}"] = float(np.nanmean(sub[f"dice3d_{x}"]))
            s[f"hd95_{x}"] = float(np.nanmean(sub[f"hd95_{x}"].replace([np.inf], np.nan)))
            s[f"nsd_{x}"] = float(np.nanmean(sub[f"nsd_{x}"]))
        s["viol_et_outside_tc"] = float(np.nanmean(sub["viol_et_outside_tc"]))
        s["viol_tc_outside_wt"] = float(np.nanmean(sub["viol_tc_outside_wt"]))
        summ[mode] = s
        print(f"[{mode}] " + " ".join(f"{x}:d3={s[f'dice3d_{x}']:.4f}" for x in REGIONS)
              + f" viol_et={s['viol_et_outside_tc']:.4f}")
    json.dump(summ, open(os.path.join(a.run_dir, f"postproc_{a.split}_{a.family}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
