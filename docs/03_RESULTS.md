# Validation 결과

## PEFT 선택

Seed 42, short prompt, DiceCE 기준

| 학습 부위 | WT | TC | ET | ET-outside-TC |
|---|---:|---:|---:|---:|
| Segmentation head | 0.842 | 0.817 | 0.769 | 7.47% |
| Encoder LoRA | 0.877 | 0.859 | 0.802 | 3.24% |
| Encoder LoRA + head | 0.889 | 0.861 | 0.810 | 3.74% |

- Head-only 대비 encoder adaptation 우위
- Combo의 평균 Dice 이득: encoder LoRA 대비 약 +0.007
- P3 고정 조건: encoder LoRA

근거: 각 P1 run 및 `p2-lora_enc-short-dicece-s42-r2`의 `summary_val_short.json`

## Boundary loss 비교

세 seed 평균±표준편차

| Loss | WT Dice3D | TC Dice3D | ET Dice3D | ET HD95 | ET NSD | ET-outside-TC |
|---|---:|---:|---:|---:|---:|---:|
| DiceCE | 0.882±0.004 | 0.861±0.003 | 0.806±0.004 | 11.262±1.057 | 0.819±0.006 | 3.059±0.467% |
| SDF | 0.883±0.003 | 0.860±0.004 | 0.805±0.002 | 11.506±1.845 | 0.821±0.003 | 2.976±0.464% |
| Boundary | 0.883±0.003 | 0.861±0.001 | 0.807±0.003 | 12.143±1.408 | 0.819±0.004 | 3.063±0.669% |

- Dice 변화: 최대 약 0.002 이내
- SDF의 ET NSD 변화: +0.002
- SDF의 ET HD95 변화: +0.243 mm 악화
- Boundary의 ET HD95 변화: +0.880 mm 악화
- SDF vs DiceCE 환자별 검정(seed 42)
  - 평균 Dice 차이: +0.000
  - Bootstrap 95% CI: [-0.003, +0.003]
  - Wilcoxon p=0.0125
  - 통계적 rank 차이와 별개로 평균 효과 크기 없음
- 결론: boundary-aware loss의 실질적 성능 향상 미확인

근거: `results/p2-lora_enc-short-dicece-*`, `results/p3-lora_enc-short-sdf-*`, `results/p3-lora_enc-short-boundary-*`

## Size tier

세 seed 평균 Dice2D

| Loss | Region | Small | Large | Small-Large gap |
|---|---|---:|---:|---:|
| DiceCE | WT | 0.613 | 0.938 | -0.325 |
| DiceCE | TC | 0.623 | 0.917 | -0.294 |
| DiceCE | ET | 0.591 | 0.906 | -0.315 |
| SDF | WT | 0.619 | 0.940 | -0.321 |
| SDF | TC | 0.624 | 0.910 | -0.287 |
| SDF | ET | 0.589 | 0.905 | -0.316 |
| Boundary | WT | 0.620 | 0.939 | -0.319 |
| Boundary | TC | 0.628 | 0.913 | -0.285 |
| Boundary | ET | 0.596 | 0.907 | -0.311 |

- Small lesion의 낮은 Dice 재현
- Kervadec boundary loss의 small-tier 변화: +0.005~0.007
- SDF의 small-tier 변화: 영역별 혼재
- 기대한 크기의 small-lesion 개선 미확인

근거: 각 run의 `summary_val_short.json` 내 `dice2d_*_{small,large}`

## Containment loss

Seed 42

| 지표 | DiceCE | Containment | 변화 |
|---|---:|---:|---:|
| WT Dice3D | 0.8775 | 0.8784 | +0.0009 |
| TC Dice3D | 0.8592 | 0.8573 | -0.0019 |
| ET Dice3D | 0.8022 | 0.8014 | -0.0008 |
| TC-outside-WT | 0.696% | 0.701% | +0.005%p |
| ET-outside-TC | 3.24% | 2.20% | -1.04%p |

- ET containment violation 감소
- WT/TC/ET Dice의 일관된 개선 없음
- 단일 seed 결과

근거: `p2-lora_enc-short-dicece-s42-r2`, `p3-lora_enc-short-containment-s42-r2`

## Hierarchy post-processing

- 집계 단위: 환자별 3D volume
  - PEFT 및 containment 표의 violation: slice별 2D 평균
  - 본 표의 대응 값: `patients_val_short.parquet`의 `viol_*_3d`를 환자별 평균
  - 같은 run이라도 집계 단위 차이로 적용 전 violation 값이 다름
- 입력 probability: 저장된 uint8 양자화본
  - 원래 evaluation summary와 미세한 차이 발생 가능

| Mode | WT Dice3D | TC Dice3D | ET Dice3D | TC-outside-WT | ET-outside-TC |
|---|---:|---:|---:|---:|---:|
| None | 0.8775 | 0.8592 | 0.8022 | 0.33% | 1.71% |
| Hard | 0.8775 | 0.8600 | 0.8028 | 0% | 0% |
| Soft | 0.8775 | 0.8600 | 0.8028 | 0% | 0% |

- Hard와 soft 결과 동일
  - Threshold 0.5에서 두 연산이 수학적으로 동일
- Hierarchy violation 완전 제거
- TC/ET Dice 소폭 상승
- 학습 단계의 constraint보다 단순하고 결정론적인 해결

근거: `results/hierarchy_postproc/postproc_val_short.json`

## nnU-Net 비교

Validation 187명, 동일 metric code

| Model | WT Dice3D | TC Dice3D | ET Dice3D | WT/TC/ET HD95 |
|---|---:|---:|---:|---|
| SAM3 encoder LoRA | 0.877 | 0.859 | 0.802 | 26.5 / 13.6 / 12.0 |
| nnU-Net 2D | 0.920 | 0.879 | 0.850 | 7.5 / 6.6 / 5.6 |
| nnU-Net 3D | 0.933 | 0.920 | 0.869 | 5.5 / 4.1 / 3.9 |

- nnU-Net 2D 대비 SAM3 평균 Dice 차이: -0.036
- Bootstrap 95% CI: [-0.045, -0.027]
- Wilcoxon p=2.61e-22
- 3D architecture의 추가 이점 확인
- 비교 조건
  - SAM3: 6 epoch PEFT, 2D slice
  - nnU-Net: 500 epoch fully supervised, 2D 또는 3D full-resolution
- 학습 예산 및 입력 차원의 차이가 혼입된 비교

근거: `results/nnunet-*`, `results/p2-lora_enc-short-dicece-s42-r2`

## 최종 판단

- 초기 가설: SDF 또는 boundary loss가 특히 ET/small lesion의 경계 성능을 개선
- 관찰 결과: Dice, HD95, NSD에서 일관되거나 실질적인 향상 없음
- 초기 가설: containment loss가 hierarchy consistency와 segmentation accuracy를 함께 개선
- 관찰 결과: violation 감소, Dice 개선 없음
- Post-processing: hierarchy consistency를 0% violation으로 보장하면서 Dice 유지
- 연구상 의미
  - 계층 일관성과 분할 정확도는 분리된 문제
  - Small lesion이 성능 병목
  - 본 설정에서는 boundary-aware objective가 병목 해결에 충분하지 않음
- 제한
  - BraTS validation 기준
  - 6 epoch, seed 3개
  - Containment는 seed 42만 수행
  - Loss weight 추가 탐색 미수행
  - Checkpoint를 동일 validation split의 quick-validation Dice로 선택
  - Validation 수치는 완전히 독립적인 held-out 추정치가 아님
  - Test set 결과 미사용
