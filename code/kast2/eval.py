"""Evaluation. Design rule: log EVERYTHING per-sample so no metric ever needs a re-run.

Outputs per (split, family) into <run_dir>/:
  slices_<split>_<family>.parquet   per-slice dice/violations/size-tiers/pixel counts
  patients_<split>_<family>.parquet per-patient 3D dice + HD95/NSD(mm) + violations
  summary_<split>_<family>.json     aggregates (also logged to wandb)
  preds_<split>_<family>/*.npz      predicted masks + prob maps (optional, save_masks=True)

Standalone CLI = P0 zero-shot / checkpoint evaluation.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from kast2.data import BraTSSliceDataset, load_split
from kast2.metrics import REGIONS, dice_per_channel, hierarchy_violation, surface_metrics_3d, size_tier
from kast2.prompts import TRAINABLE_FAMILIES, get_prompts


@torch.no_grad()
def predict_probs(model, img: torch.Tensor, family: str, dataset: str = "brats") -> torch.Tensor:
    """Sigmoid probs [B,3,H,W]; 'ensemble' averages the trainable families."""
    fams = TRAINABLE_FAMILIES if family == "ensemble" else [family]
    acc = None
    for f in fams:
        model.set_prompts(get_prompts(f, dataset))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img)
        p = torch.sigmoid(logits.float())
        acc = p if acc is None else acc + p
    return acc / len(fams)


@torch.no_grad()
def full_eval(model, data_root: str, split: str, family: str, out_dir: str,
              wandb_prefix: str | None = None, image_size: int = 512,
              save_masks: bool = False, batch_size: int = 16, native: int | None = None,
              df_override: pd.DataFrame | None = None, dataset: str = "brats") -> dict:
    device = next(model.parameters()).device
    model.eval()
    df = (df_override if df_override is not None else load_split(data_root, split))
    df = df.sort_values(["patient_id", "z"]).reset_index(drop=True)
    dl = DataLoader(BraTSSliceDataset(data_root, df, image_size), batch_size=batch_size,
                    shuffle=False, num_workers=8, pin_memory=True)
    tertiles = json.load(open(os.path.join(data_root, "tertiles.json")))

    mask_dir = os.path.join(out_dir, f"preds_{split}_{family}")
    if save_masks:
        os.makedirs(mask_dir, exist_ok=True)

    rows, pat_preds, pat_gts = [], {}, {}
    for b in dl:
        img = b["image"].to(device, non_blocking=True)
        gt_nat = b["mask_native"]                                     # [B,3,H,W] cpu
        nat = native or gt_nat.shape[-1]
        probs = predict_probs(model, img, family, dataset)
        probs_nat = F.interpolate(probs, (nat, nat), mode="bilinear",
                                  align_corners=False).cpu()
        pred_nat = (probs_nat >= 0.5).float()

        for j in range(img.shape[0]):
            pid, z = b["patient_id"][j], int(b["z"][j])
            p, g = pred_nat[j], gt_nat[j]
            d = dice_per_channel(p, g).numpy()
            row = {"patient_id": pid, "z": z, "family": family,
                   "dice_WT": d[0], "dice_TC": d[1], "dice_ET": d[2],
                   **hierarchy_violation(p)}
            for ci, r in enumerate(REGIONS):
                gp, pp = int(g[ci].sum()), int(p[ci].sum())
                row[f"gtpx_{r}"], row[f"predpx_{r}"] = gp, pp
                row[f"tier_{r}"] = size_tier(gp, tertiles[r])
            rows.append(row)
            pat_preds.setdefault(pid, {})[z] = p.numpy().astype(np.uint8)
            pat_gts.setdefault(pid, {})[z] = g.numpy().astype(np.uint8)
            if save_masks:
                np.savez_compressed(os.path.join(mask_dir, f"{pid}_{z:03d}.npz"),
                                    pred=pred_nat[j].numpy().astype(np.uint8),
                                    prob=(probs_nat[j].numpy() * 255).astype(np.uint8))

    slice_df = pd.DataFrame(rows)
    slice_df.to_parquet(os.path.join(out_dir, f"slices_{split}_{family}.parquet"))

    prows = []
    for pid, zmap in pat_preds.items():
        zs = sorted(zmap)
        pv = torch.from_numpy(np.stack([zmap[z] for z in zs], 1).astype(np.float32))
        gv = torch.from_numpy(np.stack([pat_gts[pid][z] for z in zs], 1).astype(np.float32))
        pr = {"patient_id": pid, "family": family, "n_slices": len(zs)}
        pr.update(surface_metrics_3d(pv, gv))
        d3 = dice_per_channel(pv, gv).numpy()
        pr.update({f"dice3d_{r}": d3[i] for i, r in enumerate(REGIONS)})
        pr.update({f"{k}_3d": v for k, v in hierarchy_violation(pv).items()})
        prows.append(pr)
    pat_df = pd.DataFrame(prows)
    pat_df.to_parquet(os.path.join(out_dir, f"patients_{split}_{family}.parquet"))

    summ = {"split": split, "family": family,
            "n_slices": len(slice_df), "n_patients": len(pat_df)}
    for r in REGIONS:
        summ[f"dice2d_{r}"] = float(np.nanmean(slice_df[f"dice_{r}"]))
        summ[f"dice3d_{r}"] = float(np.nanmean(pat_df[f"dice3d_{r}"]))
        summ[f"hd95_{r}"] = float(np.nanmean(pat_df[f"hd95_{r}"].replace([np.inf], np.nan)))
        summ[f"hd95_{r}_inf_frac"] = float(np.isinf(pat_df[f"hd95_{r}"]).mean())
        summ[f"nsd_{r}"] = float(np.nanmean(pat_df[f"nsd_{r}"]))
        for tier in ("small", "medium", "large"):
            sub = slice_df[slice_df[f"tier_{r}"] == tier]
            summ[f"dice2d_{r}_{tier}"] = float(np.nanmean(sub[f"dice_{r}"])) if len(sub) else float("nan")
    summ["viol_tc_outside_wt"] = float(np.nanmean(slice_df["viol_tc_outside_wt"]))
    summ["viol_et_outside_tc"] = float(np.nanmean(slice_df["viol_et_outside_tc"]))
    json.dump(summ, open(os.path.join(out_dir, f"summary_{split}_{family}.json"), "w"), indent=1)

    if wandb_prefix and wandb.run is not None:
        wandb.log({f"{wandb_prefix}/{k}": v for k, v in summ.items()
                   if isinstance(v, (int, float))})
    print(f"[eval {split}/{family}] "
          + " ".join(f"{r}:d3={summ[f'dice3d_{r}']:.3f},hd={summ[f'hd95_{r}']:.1f},"
                     f"nsd={summ[f'nsd_{r}']:.3f}" for r in REGIONS)
          + f" viol_et={summ['viol_et_outside_tc']:.3f}")
    return summ


def main():
    from kast2.sam3_seg import Sam3SemSeg
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sam3-ckpt", default="/workspace/ckpts/sam3.pt")
    ap.add_argument("--medsam3-lora", default="/workspace/ckpts/medsam3_v1_lora.pt")
    ap.add_argument("--no-medsam3-lora", action="store_true")
    ap.add_argument("--ckpt", default=None, help="trained trainable-params checkpoint")
    ap.add_argument("--peft", default="lora_enc")
    ap.add_argument("--split", default="val")
    ap.add_argument("--families", default="short,radiol,hier,ensemble")
    ap.add_argument("--dataset", default="brats", choices=["brats", "kits"])
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--subset-patients", type=int, default=0, help="0=all")
    ap.add_argument("--no-masks", action="store_true", help="skip per-slice npz dumps (I/O heavy)")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", default="/workspace/runs")
    ap.add_argument("--wandb-project", default="kast-sam3-glioma")
    ap.add_argument("--wandb-entity", default="skku-mbdl")
    a = ap.parse_args()

    run_dir = os.path.join(a.out, a.run_name)
    os.makedirs(run_dir, exist_ok=True)
    stats = json.load(open(os.path.join(a.data, "stats.json")))
    wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=a.run_name, config=vars(a))

    model = Sam3SemSeg(a.sam3_ckpt, get_prompts("short", a.dataset), a.image_size,
                       in_channels=4 if a.dataset == "brats" else 3,
                       pixel_mean=stats["mean"], pixel_std=stats["std"],
                       medsam3_lora_path=None if a.no_medsam3_lora else a.medsam3_lora,
                       peft=a.peft).cuda()
    if a.ckpt:
        missing, unexpected = model.load_state_dict(
            torch.load(a.ckpt, map_location="cuda"), strict=False)
        print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    sub = None
    if a.subset_patients:
        df = load_split(a.data, a.split)
        keep = sorted(df.patient_id.unique())[: a.subset_patients]
        sub = df[df.patient_id.isin(keep)]

    for fam in a.families.split(","):
        full_eval(model, a.data, a.split, fam, run_dir, wandb_prefix=f"{a.split}/{fam}",
                  image_size=a.image_size, save_masks=not a.no_masks,
                  df_override=sub, dataset=a.dataset)
    wandb.finish()


if __name__ == "__main__":
    main()
