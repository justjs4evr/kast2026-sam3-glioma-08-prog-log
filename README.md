# SAM3 for Glioma Sub-region Segmentation — Geometric Constraint Study (Aug 2026 snapshot)

> **Status: interim snapshot (2026-08-14), not the final study.**
> This repository archives the August 2026 stage of my research in the
> **KAST 2026 Youth Science Mentorship Program** (Korean Academy of Science and Technology),
> carried out at the **Medical Big Data Lab, Sungkyunkwan University School of Medicine**
> (mentor: Prof. Seung Won Lee; supervisors: Ryan Donghan Kwon, Hyunwoo Yang).
> The project later evolved into a different, preregistered study on prompt-induced
> segmentation failures; that later work is **not** included here.

## What this stage studied

We fine-tuned **SAM3** (text-promptable segmentation foundation model, with MedSAM3 medical
prior) on **BraTS 2021** brain-tumor MRI to segment the nested glioma sub-regions
(ET ⊂ TC ⊂ WT), and tested whether **geometry-aware objectives** improve accuracy:

- Parameter-efficient fine-tuning (LoRA, rank 12) of the SAM3 vision encoder, with the patch
  embedding extended from 3 to 4 MRI channels (T1, T1ce, T2, FLAIR) and position/RoPE tables
  re-interpolated
- Loss comparison: **DiceCE** vs **SDF-weighted** vs **Kervadec boundary loss** vs
  **containment (hierarchy) loss**, 3 seeds each (containment: 1 seed)
- **Hierarchy post-processing** (hard/soft intersection) to enforce ET ⊂ TC ⊂ WT
- **nnU-Net v2** 2D / 3D baselines (500 epochs) with the same split and metric code
- Metrics: 3D/2D Dice, HD95, NSD, hierarchy-violation rate, lesion-size tiers;
  patient-level bootstrap CIs and Wilcoxon tests

## Key results (validation, 187 patients)

| Loss (3 seeds, mean ± sd) | WT Dice | TC Dice | ET Dice | ET HD95 |
|---|---:|---:|---:|---:|
| DiceCE | 0.882 ± 0.004 | 0.861 ± 0.003 | 0.806 ± 0.004 | 11.26 ± 1.06 |
| SDF-weighted | 0.883 ± 0.003 | 0.860 ± 0.004 | 0.805 ± 0.002 | 11.51 ± 1.85 |
| Kervadec boundary | 0.883 ± 0.003 | 0.861 ± 0.001 | 0.807 ± 0.003 | 12.14 ± 1.41 |

| Model (seed 42) | WT | TC | ET |
|---|---:|---:|---:|
| SAM3 + encoder LoRA (6 epochs) | 0.877 | 0.859 | 0.802 |
| nnU-Net 2D (500 epochs) | 0.920 | 0.879 | 0.850 |
| nnU-Net 3D full-res (500 epochs) | 0.933 | 0.920 | 0.869 |

**Findings (honest negative result):**
- Boundary-aware losses changed Dice by ≤ 0.002 — within seed-to-seed variation.
- Containment loss reduced hierarchy violations but did not improve Dice; simple
  post-processing removed violations entirely (0%) at no Dice cost.
- Small lesions (Dice ≈ 0.6 vs ≈ 0.9 for large) are the real bottleneck.
- During cross-evaluation we noticed that **changing the prompt wording moved Dice far more
  than any loss function** — the observation that led to the next stage of the project.

Full tables, statistics and limitations: [`docs/03_RESULTS.md`](docs/03_RESULTS.md) (Korean).

## Repository layout

```text
code/kast2/        SAM3 wrapper (4-ch patch embed, LoRA), dataset, losses, train/eval, metrics
code/scripts/      BraTS 2021 preprocessing, hierarchy post-processing, nnU-Net convert/eval
data_meta/         patient-level split (seed 42; slice index only, no images)
docs/              research-note continuation, methods & run list, results (Korean)
results/           per-run config.json, summary json, per-patient parquet metrics
```

`code/` is copied unchanged from the lab working repository, so it also contains features
added after this stage (paraphrase prompts, KiTS23 option, cross-prompt evaluation); those
paths were not used for the results here.

## Reproducing (outline)

```bash
pip install -r requirements.txt          # plus SAM3 / MedSAM3 from source
python code/scripts/preprocess_brats21.py   # BraTS 2021 → 240×240 axial .npz slices
python -m kast2.train --data <brats21_dir> --peft lora_enc --prompt-family short \
    --loss dicece --seed 42 --run-name p2-lora_enc-short-dicece-s42-r2
python -m kast2.eval  ...                # see docs/02_METHODS_AND_RUNS.md
```

Training used single NVIDIA A100 (40/80 GB) GPUs. Paths default to `/workspace/...`.

## Not included

Model checkpoints, prediction masks, MRI images (BraTS data must be obtained under its own
license), test-set results, and all later-stage (prompt robustness / failure decomposition)
code and results.

## Data & credits

- BraTS 2021: Baid U. et al., *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark*, arXiv:2107.02314.
- SAM 3: Carion N. et al., arXiv:2511.16719. MedSAM3: Liu A. et al., arXiv:2511.19046.
- nnU-Net: Isensee F. et al., *Nature Methods* 18, 2021.

Code and documents © 2026 Doyoun Choi, Ryan Donghan Kwon, Hyunwoo Yang
(Medical Big Data Lab, SKKU). Shared for review purposes; not licensed for reuse.
