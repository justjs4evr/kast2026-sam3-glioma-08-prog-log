"""Losses. All treat the 3 channels (WT/TC/ET) as independent foreground sigmoid heads.

  dicece   : MONAI DiceCE, include_background=True (all channels are foreground)
  sdf      : SDF-weighted BCE (FocusSDF-style pixel weighting) + Dice
  boundary : Kervadec boundary loss (MedIA 2021) + DiceCE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss


def _dicece(lambda_ce: float = 1.0) -> DiceCELoss:
    return DiceCELoss(sigmoid=True, include_background=True, squared_pred=True,
                      reduction="mean", batch=True, lambda_ce=lambda_ce)


class DiceCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = _dicece()

    def forward(self, logits, target, sdf=None):
        return self.loss(logits, target)


class SDFWeightedDiceCE(nn.Module):
    """Pixel-weighted BCE: w=1 inside & on boundary, exp(-gamma*d) decaying outside; + Dice."""

    def __init__(self, gamma: float = 0.02, lam_bce: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.lam = lam_bce
        self.dice = _dicece(lambda_ce=0.0)

    def forward(self, logits, target, sdf=None):
        assert sdf is not None, "loss=sdf requires per-pixel signed distances"
        w = torch.where(sdf <= 0, torch.ones_like(sdf), torch.exp(-self.gamma * sdf))
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="mean")
        return self.dice(logits, target) + self.lam * bce


class KervadecBoundary(nn.Module):
    """L = DiceCE + w * mean(phi_G · sigmoid(logits)), phi_G = signed distance to GT boundary."""

    def __init__(self, w_boundary: float = 0.01):
        super().__init__()
        self.w = w_boundary
        self.dice = _dicece()

    def forward(self, logits, target, sdf=None):
        assert sdf is not None, "loss=boundary requires per-pixel signed distances"
        return self.dice(logits, target) + self.w * (sdf * torch.sigmoid(logits)).mean()


class ContainmentDiceCE(nn.Module):
    """DiceCE + pixelwise hierarchy containment penalty:
    L_contain = mean(relu(P_ET - P_TC) + relu(P_TC - P_WT)).
    First explicit containment loss in the text-prompted regime (channels = WT,TC,ET)."""

    def __init__(self, w_contain: float = 0.05):
        super().__init__()
        self.w = w_contain
        self.dice = _dicece()

    def forward(self, logits, target, sdf=None):
        p = torch.sigmoid(logits)
        contain = (torch.relu(p[:, 2] - p[:, 1]) + torch.relu(p[:, 1] - p[:, 0])).mean()
        return self.dice(logits, target) + self.w * contain


def build_loss(name: str, **kw) -> nn.Module:
    return {"dicece": DiceCE, "sdf": SDFWeightedDiceCE, "boundary": KervadecBoundary,
            "containment": ContainmentDiceCE}[name](**kw)


def loss_needs_sdf(name: str) -> bool:
    return name in ("sdf", "boundary")
