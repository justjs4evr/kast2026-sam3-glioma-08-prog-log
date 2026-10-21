#!/usr/bin/env python3
"""BraTS2021 (Kaggle layout) -> nnU-Net v2 Dataset137 with region-based labels.

Labels: BraTS {0,1(NCR),2(ED),4(ET)} -> nnUNet {0, 1=ED, 2=NCR, 3=ET}
Regions: WT=(1,2,3), TC=(2,3), ET=(3,), regions_class_order=(1,2,3)  [official Dataset137 scheme]
Our test-split patients go to imagesTs/labelsTs (never seen by nnU-Net training).
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk


def convert_label(src: str, dst: str):
    img = sitk.ReadImage(src)
    a = sitk.GetArrayFromImage(img)
    out = np.zeros_like(a)
    out[a == 2] = 1
    out[a == 1] = 2
    out[a == 4] = 3
    o = sitk.GetImageFromArray(out)
    o.CopyInformation(img)
    sitk.WriteImage(o, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir containing BraTS2021_* patient folders")
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    split = pd.read_csv(a.split_csv).drop_duplicates("patient_id").set_index("patient_id")["split"]
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    # find patient dirs (possibly nested one level)
    pdirs = []
    for root, dirs, _ in os.walk(a.raw):
        pdirs += [os.path.join(root, d) for d in dirs if d.startswith("BraTS2021_")]
        if pdirs:
            break
    if not pdirs:
        for root, dirs, _ in os.walk(a.raw):
            pdirs += [os.path.join(root, d) for d in dirs if d.startswith("BraTS2021_")]
    assert pdirs, f"no BraTS2021_* dirs under {a.raw}"

    mods = ["t1", "t1ce", "t2", "flair"]
    n_tr = n_ts = n_skip = 0
    for pd_ in sorted(pdirs):
        pid = os.path.basename(pd_)
        if pid not in split.index:
            n_skip += 1
            continue
        is_test = split.loc[pid] == "test"
        img_dir = "imagesTs" if is_test else "imagesTr"
        lab_dir = "labelsTs" if is_test else "labelsTr"
        ok = True
        for i, m in enumerate(mods):
            src = os.path.join(pd_, f"{pid}_{m}.nii.gz")
            if not os.path.exists(src):
                ok = False
                break
            dst = os.path.join(a.out, img_dir, f"{pid}_{i:04d}.nii.gz")
            if not os.path.exists(dst):
                shutil.copy(src, dst)
        seg = os.path.join(pd_, f"{pid}_seg.nii.gz")
        if not ok or not os.path.exists(seg):
            n_skip += 1
            continue
        dst = os.path.join(a.out, lab_dir, f"{pid}.nii.gz")
        if not os.path.exists(dst):
            convert_label(seg, dst)
        if is_test:
            n_ts += 1
        else:
            n_tr += 1

    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    generate_dataset_json(
        a.out,
        channel_names={0: "T1", 1: "T1ce", 2: "T2", 3: "Flair"},
        labels={"background": 0, "whole_tumor": (1, 2, 3), "tumor_core": (2, 3), "enhancing_tumor": (3,)},
        num_training_cases=n_tr,
        file_ending=".nii.gz",
        regions_class_order=(1, 2, 3),
        dataset_name="Dataset137_BraTS2021",
        license="BraTS2021 (Kaggle mirror)",
    )
    print(f"converted: train+val={n_tr} test={n_ts} skipped={n_skip}")


if __name__ == "__main__":
    main()
