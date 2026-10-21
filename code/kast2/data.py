"""BraTS-2021 2D slice dataset.

On-disk format (produced by scripts/preprocess_brats21.py):
  <root>/slices/<patient>/<patient>_zZZZ.npz  with:
      image: uint8 [4, 240, 240]  (T1, T1ce, T2, FLAIR; per-volume 1-99 percentile -> 0-255)
      mask : uint8 [3, 240, 240]  (WT, TC, ET binary)
  <root>/split_seed42.csv : patient-level 70/15/15 split (columns:
      filepath, patient_id, z, split, has_tumor, px_WT, px_TC, px_ET)
  <root>/stats.json    : per-channel mean/std of image/255 over train split
  <root>/tertiles.json : per-region GT-pixel tertile boundaries (size-tier analysis)
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


def sdf_from_mask(mask: np.ndarray) -> np.ndarray:
    """Signed distance per channel (negative inside, positive outside). mask: [C,H,W] binary."""
    out = np.empty(mask.shape, dtype=np.float32)
    for c in range(mask.shape[0]):
        m = mask[c] > 0
        if m.any():
            out[c] = distance_transform_edt(~m) - distance_transform_edt(m)
        else:
            out[c] = distance_transform_edt(np.ones_like(m))
    return out


class BraTSSliceDataset(Dataset):
    def __init__(self, root: str, df: pd.DataFrame, image_size: int = 512,
                 augment: bool = False, return_sdf: bool = False, native: int | None = None):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.return_sdf = return_sdf
        self.native = native

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        d = np.load(os.path.join(self.root, row.filepath))
        img = torch.from_numpy(d["image"].astype(np.float32) / 255.0)  # [C,H,W] in [0,1]
        if img.shape[0] == 1:                                          # CT: replicate to RGB
            img = img.repeat(3, 1, 1)
        msk = torch.from_numpy(d["mask"].astype(np.float32))           # [3,H,W]

        if self.augment:  # joint flips/rot90 (image+mask together)
            if torch.rand(()) < 0.5:
                img, msk = torch.flip(img, [-1]), torch.flip(msk, [-1])
            if torch.rand(()) < 0.5:
                img, msk = torch.flip(img, [-2]), torch.flip(msk, [-2])
            k = int(torch.randint(0, 4, ()))
            if k:
                img, msk = torch.rot90(img, k, [-2, -1]), torch.rot90(msk, k, [-2, -1])

        s = self.image_size
        out = {
            "image": F.interpolate(img[None], (s, s), mode="bilinear", align_corners=False)[0],
            "mask": F.interpolate(msk[None], (s, s), mode="nearest")[0],
            "mask_native": msk,
            "patient_id": row.patient_id,
            "z": int(row.z),
        }
        if self.return_sdf:
            sdf = torch.from_numpy(sdf_from_mask(msk.numpy().astype(np.uint8)))
            native = self.native or msk.shape[-1]
            out["sdf"] = F.interpolate(sdf[None], (s, s), mode="bilinear",
                                       align_corners=False)[0] * (s / native)
        return out


def load_split(root: str, split: str, split_csv: str = "split_seed42.csv") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(root, split_csv))
    return df[df.split == split].copy()


def tumor_biased_weights(df: pd.DataFrame, empty_weight: float = 0.15) -> torch.Tensor:
    """Sampling weights: tumor slices 1.0, empty slices `empty_weight`.
    (Uniform sampling over all slices lets ~40% empty slices teach 'predict nothing'.)"""
    w = np.where(df.has_tumor.to_numpy(), 1.0, empty_weight)
    return torch.from_numpy(w.astype(np.float64))
