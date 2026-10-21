# 실험 방법 및 run 목록

> `code/`에는 저장소의 실험 파일을 수정 없이 복사했기 때문에 이후 추가된 paraphrase, KiTS23,
> cross-prompt evaluation 기능도 포함. 본 문서의 실험에서는 해당 기능 미사용. 사용 범위는
> BraTS 2021, short prompt, geometry-aware loss 및 hierarchy post-processing으로 한정

## 데이터

- Dataset: BraTS 2021 training set 1,251명
- 입력 modality: T1, T1ce, T2, FLAIR
- 정답 영역
  - WT = label 1 ∪ 2 ∪ 4
  - TC = label 1 ∪ 4
  - ET = label 4
- Split: patient-level 70/15/15, seed 42
  - Train 875명
  - Validation 187명
  - Test 189명, 본 패키지 결과에서 미사용
- 전처리
  - Volume/modality별 foreground z-score
  - [-5, 5] clipping 후 uint8 변환
  - Axial 240×240 slice 저장
  - Train: tumor-containing slice만 사용
  - Validation: brain-containing slice 전체 사용
  - 학습 시 512×512 resize
- 관련 파일
  - `code/scripts/preprocess_brats21.py`
  - `data_meta/split_seed42.csv`

## 모델

- Base model: SAM3
- Initial medical prior: MedSAM3-v1 vision encoder LoRA merge
- Output: `semantic_seg` head
- MRI 입력 수정
  - Patch embedding 3채널 → 4채널
  - 네 번째 채널 가중치: 기존 RGB 가중치 평균
  - Position embedding 보간
  - RoPE table 재계산
- PEFT 기본 설정
  - Vision trunk의 qkv/proj/fc1/fc2에 LoRA
  - Rank 12, alpha 24, dropout 0.05
  - Patch embedding 학습
- 관련 파일
  - `code/kast2/sam3_seg.py`
  - `code/kast2/train.py`

## 학습 설정

| 항목 | 설정 |
|---|---|
| Prompt | short: `glioma`, `tumor core`, `enhancing tumor` |
| Epoch | 6 |
| Samples per epoch | 8,000 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Scheduler | OneCycle, pct_start 0.06 |
| Precision | bf16 autocast |
| Image size | 512×512 |
| Model selection | 매 epoch validation tumor slice 1,200장 quick validation |
| Seeds | 42, 2026, 67 |
| GPU | A100 SXM4 80GB 또는 40GB |

```bash
python -m kast2.train \
  --data /workspace/data/brats21 \
  --peft lora_enc \
  --prompt-family short \
  --loss dicece \
  --seed 42 \
  --run-name p2-lora_enc-short-dicece-s42-r2
```

Loss 비교 시 `--loss`를 `sdf`, `boundary`, `containment`로 변경

## Loss 정의

### DiceCE

- MONAI `DiceCELoss`
- Sigmoid output
- 세 채널 모두 foreground channel
- `squared_pred=True`, batch reduction

### SDF-weighted DiceCE

- Signed distance
  - 내부: 음수
  - 외부: 양수
- Pixel weight
  - 내부 및 경계: 1
  - 외부: `exp(-0.02 × distance)`
- 최종 loss: Dice + weighted BCE
- 구현: `code/kast2/losses.py`, `code/kast2/data.py`

### Kervadec boundary loss

- 최종 loss: DiceCE + `0.01 × mean(SDF × sigmoid(logits))`
- 구현: `code/kast2/losses.py`

### Containment loss

- 계층 관계: ET ⊂ TC ⊂ WT
- Penalty: `mean(ReLU(P_ET-P_TC) + ReLU(P_TC-P_WT))`
- Weight: 0.05
- 최종 loss: DiceCE + containment penalty
- 구현: `code/kast2/losses.py`

## Hierarchy post-processing

- Hard
  - `TC = TC ∩ WT`
  - `ET = ET ∩ TC`
- Soft
  - `P_TC = min(P_TC, P_WT)`
  - `P_ET = min(P_ET, P_TC)`
  - 이후 threshold 0.5
- 구현: `code/scripts/hierarchy_postproc.py`

## 평가

- Validation 187명 전체
- Slice prediction을 환자별 3D volume으로 재구성
- 지표
  - Dice3D
  - HD95(mm)
  - NSD, tolerance 1 mm
  - TC-outside-WT 비율
  - ET-outside-TC 비율
  - Size-tier Dice2D
- Empty GT/pred 처리
  - 둘 다 empty: NaN 후 평균 제외
  - 한쪽만 empty: HD95 inf, NSD 0
- 관련 파일
  - `code/kast2/eval.py`
  - `code/kast2/metrics.py`

## Size tier

- 기준: train split의 tumor-positive slice별 GT pixel count
- 영역별 P33.33/P66.67 사용

| Region | P33.33 | P66.67 | Non-empty slices |
|---|---:|---:|---:|
| WT | 724.4337 | 1887 | 56,990 |
| TC | 386 | 1022 | 37,585 |
| ET | 277 | 637 | 35,764 |

## Run 목록

| 구분 | Run |
|---|---|
| Head-only PEFT | `p1-seg_head_ft-short-dicece-s42-r2` |
| Encoder LoRA + head | `p1-lora_enc+seg_head-short-dicece-s42-r2` |
| DiceCE | `p2-lora_enc-short-dicece-s{42,2026,67}-r2` |
| SDF | `p3-lora_enc-short-sdf-s{42,2026,67}-r2` |
| Boundary | `p3-lora_enc-short-boundary-s{42,2026,67}-r2` |
| Containment | `p3-lora_enc-short-containment-s42-r2` |
| Post-processing | `results/hierarchy_postproc/` |
| nnU-Net 2D | `results/nnunet-2d-500ep/` |
| nnU-Net 3D | `results/nnunet-3d-500ep/` |

## nnU-Net

- Dataset137 region-based label scheme
- 본 실험의 train/validation split을 `splits_final.json`에 적용
- Config: `2d`, `3d_fullres`
- Trainer: `nnUNetTrainer_500epochs`, fold 0
- GPU: H100 NVL 94GB
- SAM3와 같은 `metrics.py`로 재평가
- 관련 파일
  - `code/scripts/nnunet_convert.py`
  - `code/scripts/nnunet_baseline.sh`
  - `code/scripts/nnunet_eval.py`
