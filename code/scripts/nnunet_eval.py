#!/usr/bin/env python3
"""Evaluate nnU-Net predictions with OUR metrics (3D dice / HD95 / NSD / violations).

nnU-Net label map {0,1=ED,2=NCR,3=ET} -> regions WT={1,2,3}, TC={2,3}, ET={3}.
Outputs patients_*.parquet + summary json in --out (same schema as kast2 evals).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

sys.path.insert(0, "/workspace/kast")
from kast2.metrics import REGIONS, dice_per_channel, hierarchy_violation, surface_metrics_3d


def to_regions(a: np.ndarray) -> np.ndarray:
    wt = np.isin(a, (1, 2, 3)).astype(np.float32)
    tc = np.isin(a, (2, 3)).astype(np.float32)
    et = (a == 3).astype(np.float32)
    return np.stack([wt, tc, et])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for pf in sorted(glob.glob(os.path.join(a.pred, "*.nii.gz"))):
        pid = os.path.basename(pf).replace(".nii.gz", "")
        gf = os.path.join(a.gt, f"{pid}.nii.gz")
        if not os.path.exists(gf):
            continue
        p = to_regions(sitk.GetArrayFromImage(sitk.ReadImage(pf)))
        g = to_regions(sitk.GetArrayFromImage(sitk.ReadImage(gf)))
        pv, gv = torch.from_numpy(p), torch.from_numpy(g)
        r = {"patient_id": pid}
        d3 = dice_per_channel(pv, gv).numpy()
        r.update({f"dice3d_{x}": d3[i] for i, x in enumerate(REGIONS)})
        r.update(surface_metrics_3d(pv, gv))
        r.update({f"{k}_3d": v for k, v in hierarchy_violation(pv).items()})
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(a.out, "patients_val_nnunet.parquet"))
    summ = {"n_patients": len(df)}
    for x in REGIONS:
        summ[f"dice3d_{x}"] = float(np.nanmean(df[f"dice3d_{x}"]))
        summ[f"hd95_{x}"] = float(np.nanmean(df[f"hd95_{x}"].replace([np.inf], np.nan)))
        summ[f"nsd_{x}"] = float(np.nanmean(df[f"nsd_{x}"]))
    summ["viol_et_outside_tc"] = float(np.nanmean(df["viol_et_outside_tc_3d"]))
    json.dump(summ, open(os.path.join(a.out, "summary_val_nnunet.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
