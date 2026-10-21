"""Training entrypoint.

Example:
  PYTHONPATH=/workspace/kast:/workspace/MedSAM3 python -m kast2.train \
      --data /workspace/data/brats21 --peft lora_enc --prompt-family short \
      --loss dicece --seed 42 --run-name p2-lora_enc-short-dicece-s42-r2
"""
import argparse
import json
import os
import random
import subprocess
import time

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler

from kast2.data import BraTSSliceDataset, load_split, tumor_biased_weights
from kast2.eval import full_eval
from kast2.losses import build_loss, loss_needs_sdf
from kast2.metrics import dice_per_channel
from kast2.prompts import EVAL_FAMILIES, get_prompts
from kast2.sam3_seg import Sam3SemSeg


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return "unknown"


def auto_batch_size() -> int:
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    return 12 if vram > 70 else (8 if vram > 44 else 4)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sam3-ckpt", default="/workspace/ckpts/sam3.pt")
    ap.add_argument("--medsam3-lora", default="/workspace/ckpts/medsam3_v1_lora.pt")
    ap.add_argument("--no-medsam3-lora", action="store_true")
    ap.add_argument("--peft", default="lora_enc",
                    choices=["lora_enc", "seg_head_ft", "lora_enc+seg_head"])
    ap.add_argument("--prompt-family", default="short")
    ap.add_argument("--dataset", default="brats", choices=["brats", "kits"])
    ap.add_argument("--loss", default="dicece", choices=["dicece", "sdf", "boundary", "containment"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--samples-per-epoch", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto by VRAM")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=12)
    ap.add_argument("--empty-weight", type=float, default=0.15,
                    help="sampling weight of tumor-free slices (tumor slices = 1.0)")
    ap.add_argument("--quickval-n", type=int, default=1200)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", default="/workspace/runs")
    ap.add_argument("--wandb-project", default="kast-sam3-glioma")
    ap.add_argument("--wandb-entity", default="skku-mbdl")
    ap.add_argument("--eval-families", default="ALL",
                    help="comma list or ALL — cross-family eval after training")
    return ap.parse_args()


@torch.no_grad()
def quickval(model, dl, device) -> tuple[np.ndarray, float]:
    model.eval()
    dices = []
    for b in dl:
        img = b["image"].to(device, non_blocking=True)
        msk = b["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(img)
        pred = (torch.sigmoid(logits.float()) >= 0.5).float()
        for j in range(pred.shape[0]):
            dices.append(dice_per_channel(pred[j], msk[j]).cpu().numpy())
    per_region = np.nanmean(np.stack(dices), axis=0)
    return per_region, float(np.nanmean(per_region))


def main():
    a = parse_args()
    set_seed(a.seed)
    device = torch.device("cuda")
    if a.batch_size == 0:
        a.batch_size = auto_batch_size()

    run_dir = os.path.join(a.out, a.run_name)
    os.makedirs(run_dir, exist_ok=True)
    cfg = vars(a) | {"git_commit": git_commit(), "gpu": torch.cuda.get_device_name(0),
                     "codebase": "kast2"}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    wandb.init(project=a.wandb_project, entity=a.wandb_entity, name=a.run_name, config=cfg)

    stats = json.load(open(os.path.join(a.data, "stats.json")))
    need_sdf = loss_needs_sdf(a.loss)

    tr_df = load_split(a.data, "train")
    va_df = load_split(a.data, "val")
    va_sub = va_df[va_df.has_tumor].sample(
        n=min(a.quickval_n, int(va_df.has_tumor.sum())), random_state=0)

    train_ds = BraTSSliceDataset(a.data, tr_df, a.image_size, augment=True, return_sdf=need_sdf)
    sampler = WeightedRandomSampler(tumor_biased_weights(tr_df, a.empty_weight),
                                    num_samples=a.samples_per_epoch, replacement=False)
    tl = DataLoader(train_ds, batch_size=a.batch_size, sampler=sampler, num_workers=8,
                    pin_memory=True, persistent_workers=True, drop_last=True)
    vl = DataLoader(BraTSSliceDataset(a.data, va_sub, a.image_size),
                    batch_size=a.batch_size * 2, shuffle=False, num_workers=6, pin_memory=True)

    in_ch = 4 if a.dataset == "brats" else 3
    model = Sam3SemSeg(a.sam3_ckpt, get_prompts(a.prompt_family, a.dataset), a.image_size,
                       in_channels=in_ch,
                       pixel_mean=stats["mean"], pixel_std=stats["std"],
                       medsam3_lora_path=None if a.no_medsam3_lora else a.medsam3_lora,
                       peft=a.peft, lora_r=a.lora_r).to(device)
    criterion = build_loss(a.loss)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps_per_epoch = a.samples_per_epoch // (a.batch_size * a.grad_accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.06)

    best_dice, gstep = -1.0, 0
    for ep in range(a.epochs):
        model.train()
        model.set_prompts(get_prompts(a.prompt_family, a.dataset))
        t0, ep_loss, nb = time.time(), 0.0, 0
        for i, b in enumerate(tl):
            img = b["image"].to(device, non_blocking=True)
            msk = b["mask"].to(device, non_blocking=True)
            sdf = b["sdf"].to(device, non_blocking=True) if need_sdf else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(img)
            loss = criterion(logits.float(), msk, sdf)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at ep{ep} step{i}")
            (loss / a.grad_accum).backward()
            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
            ep_loss += loss.item()
            nb += 1
            if i % 50 == 0:
                wandb.log({"train/loss": loss.item(),
                           "train/lr": sched.get_last_lr()[0]}, step=gstep)

        per_region, avg = quickval(model, vl, device)
        wandb.log({"val/dice_WT": per_region[0], "val/dice_TC": per_region[1],
                   "val/dice_ET": per_region[2], "val/dice_avg": avg, "epoch": ep,
                   "train/epoch_loss": ep_loss / max(nb, 1),
                   "train/epoch_sec": time.time() - t0}, step=gstep)
        print(f"[ep{ep}] loss={ep_loss/max(nb,1):.4f} "
              f"dice WT/TC/ET={per_region.round(4)} avg={avg:.4f}", flush=True)

        sd = model.trainable_state_dict()
        torch.save(sd, os.path.join(run_dir, "last.pt"))
        if avg > best_dice:
            best_dice = avg
            torch.save(sd, os.path.join(run_dir, "best.pt"))
            wandb.run.summary["best_quickval_dice"] = best_dice

    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"),
                                     map_location=device), strict=False)
    fams = EVAL_FAMILIES if a.eval_families == "ALL" else a.eval_families.split(",")
    for fam in fams:
        full_eval(model, a.data, "val", fam, run_dir, wandb_prefix=f"fullval/{fam}",
                  image_size=a.image_size, save_masks=(fam == a.prompt_family),
                  dataset=a.dataset)
    wandb.finish()


if __name__ == "__main__":
    main()
