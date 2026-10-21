"""Metrics: Dice (2D/3D), HD95 + NSD in mm (3D, MONAI), hierarchy-violation rates, size tiers."""
import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance, compute_surface_dice

REGIONS = ["WT", "TC", "ET"]


def dice_per_channel(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """pred, gt: [C, ...] binary. Returns [C]; NaN where both are empty (excluded from means)."""
    dims = tuple(range(1, pred.dim()))
    inter = (pred * gt).sum(dims)
    denom = pred.sum(dims) + gt.sum(dims)
    d = (2 * inter + eps) / (denom + eps)
    d[denom == 0] = float("nan")
    return d


def hierarchy_violation(pred: torch.Tensor) -> dict:
    """Fraction of predicted TC outside predicted WT / predicted ET outside predicted TC.
    pred: [3, ...] binary (WT, TC, ET). NaN when the child region is empty."""
    wt, tc, et = pred[0], pred[1], pred[2]
    tc_n, et_n = tc.sum().item(), et.sum().item()
    return {
        "viol_tc_outside_wt": float((tc * (1 - wt)).sum().item() / tc_n) if tc_n else float("nan"),
        "viol_et_outside_tc": float((et * (1 - tc)).sum().item() / et_n) if et_n else float("nan"),
    }


def surface_metrics_3d(pred_vol: torch.Tensor, gt_vol: torch.Tensor,
                       spacing=(1.0, 1.0, 1.0), nsd_tol_mm: float = 1.0) -> dict:
    """HD95 / NSD per region on native-grid volumes. pred_vol, gt_vol: [3, D, H, W] binary.
    Conventions: both empty -> NaN (skip); exactly one empty -> HD95=inf, NSD=0."""
    res = {}
    p, g = pred_vol.unsqueeze(0), gt_vol.unsqueeze(0)
    for ci, r in enumerate(REGIONS):
        pc, gc = p[:, ci:ci + 1], g[:, ci:ci + 1]
        if gc.sum() == 0 and pc.sum() == 0:
            res[f"hd95_{r}"], res[f"nsd_{r}"] = float("nan"), float("nan")
        elif gc.sum() == 0 or pc.sum() == 0:
            res[f"hd95_{r}"], res[f"nsd_{r}"] = float("inf"), 0.0
        else:
            hd = compute_hausdorff_distance(pc, gc, percentile=95, spacing=list(spacing))
            sd = compute_surface_dice(pc, gc, class_thresholds=[nsd_tol_mm], spacing=list(spacing))
            res[f"hd95_{r}"], res[f"nsd_{r}"] = float(hd.item()), float(sd.item())
    return res


def size_tier(gt_pixels: int, tertiles: dict) -> str:
    if gt_pixels <= 0:
        return "none"
    if gt_pixels <= tertiles["P33.33"]:
        return "small"
    if gt_pixels <= tertiles["P66.67"]:
        return "medium"
    return "large"
