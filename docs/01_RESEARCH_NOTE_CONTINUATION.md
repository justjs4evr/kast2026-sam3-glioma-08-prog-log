# 기존 연구노트 이후 추가 실험

## 연구 주제

- 가제: `Hierarchy- and Boundary-Aware Text-Promptable Segmentation of Glioma Sub-regions with SAM3`
- 대상 영역: whole tumor(WT), tumor core(TC), enhancing tumor(ET)
- 구조적 관계: ET ⊂ TC ⊂ WT
- 초기 목표
  - Boundary-aware supervision을 통한 경계 정확도 개선
  - Hierarchical constraint를 통한 해부학적으로 불가능한 예측 감소
  - Small lesion에서 더 큰 개선 확인

## 실험 진행

### 1. 학습 기반선 결정

- SAM3의 text-conditioned semantic segmentation head 사용
- BraTS 네 modality 입력을 위한 4-channel patch embedding 적용
- 비교한 parameter-efficient fine-tuning 방식
  - Segmentation head fine-tuning
  - Vision encoder LoRA
  - Vision encoder LoRA + segmentation head fine-tuning
- 결과
  - Head-only Dice3D: 0.842 / 0.817 / 0.769
  - Encoder LoRA Dice3D: 0.877 / 0.859 / 0.802
  - Encoder LoRA + head Dice3D: 0.889 / 0.861 / 0.810
- 이후 loss 비교의 고정 설정: encoder LoRA + short text prompt

### 2. Boundary-aware loss 비교

- 기준 loss: DiceCE
- 추가 조건
  - SDF-weighted DiceCE
  - Kervadec boundary loss + DiceCE
- Seed: 42, 2026, 67
- 평가: 187명 validation, 환자별 3D 재구성
- 세 seed 평균 Dice3D
  - DiceCE: 0.882 / 0.861 / 0.806
  - SDF: 0.883 / 0.860 / 0.805
  - Boundary: 0.883 / 0.861 / 0.807
- ET HD95
  - DiceCE: 11.262 mm
  - SDF: 11.506 mm
  - Boundary: 12.143 mm
- ET NSD
  - DiceCE: 0.819
  - SDF: 0.821
  - Boundary: 0.819
- 판정: boundary-aware loss의 일관된 성능 개선 없음

### 3. Lesion size별 확인

- Train split의 영역별 pixel count로 small/medium/large tertile 고정
- DiceCE 기준 세 seed 평균
  - WT: small 0.613 / large 0.938
  - TC: small 0.623 / large 0.917
  - ET: small 0.591 / large 0.906
- Kervadec boundary loss의 small-tier 변화
  - WT: +0.007
  - TC: +0.005
  - ET: +0.006
- 해석
  - Small lesion이 주요 오차 구간임은 확인
  - Boundary loss의 small-tier 이득은 0.005–0.007 수준
  - 3D Dice 및 surface metric 개선으로 연결되지 않음

### 4. Hierarchical constraint 비교

- 학습 시 containment penalty 추가
  - `ReLU(P_ET - P_TC) + ReLU(P_TC - P_WT)`
- Seed 42 결과
  - ET-outside-TC: 3.24% → 2.20%
  - Dice3D WT: 0.877 → 0.878
  - Dice3D TC: 0.859 → 0.857
  - Dice3D ET: 0.802 → 0.801
- 판정: violation 감소, Dice 개선 없음

### 5. Hierarchy post-processing

- 저장된 probability map에 hard/soft intersection 적용
- 집계 단위: 환자별 3D volume
  - 앞 containment 결과의 violation은 slice별 2D 평균
  - 집계 단위 차이로 적용 전 violation 값이 서로 다름
- 저장 probability: uint8 양자화본 사용
  - 원래 평가 summary와 미세한 수치 차이 발생 가능
- ET-outside-TC: 1.71% → 0%
- TC-outside-WT: 0.33% → 0%
- Dice3D
  - TC: 0.8592 → 0.8600
  - ET: 0.8022 → 0.8028
- 판정
  - 학습 loss보다 단순 intersection이 hierarchy consistency를 더 확실하게 보장
  - Dice 손실 없음

### 6. nnU-Net과 비교

- 동일한 환자 split과 동일한 평가 코드 사용
- 학습 조건 차이
  - SAM3: 6 epoch PEFT, 2D slice
  - nnU-Net: 500 epoch fully supervised, 2D 또는 3D full-resolution
- Validation Dice3D
  - SAM3 encoder LoRA: 0.877 / 0.859 / 0.802
  - nnU-Net 2D: 0.920 / 0.879 / 0.850
  - nnU-Net 3D: 0.933 / 0.920 / 0.869
- 판정: 해당 학습 조건에서 nnU-Net이 overlap 및 surface metric에서 우위
- 해석 제한: 학습 예산과 입력 차원이 다른 비교

## 연구노트용 결론

- Boundary-aware supervision의 가설
  - Small lesion 및 ET boundary에서 Dice, HD95, NSD 개선 예상
  - 세 seed 실험에서 기대한 규모의 개선 미확인
- Hierarchical constraint의 가설
  - Containment loss로 violation 감소 확인
  - Segmentation accuracy 개선으로는 연결되지 않음
- 후처리 결과
  - 계층 일관성 문제는 학습 loss 없이도 intersection으로 해결 가능
- 전체 판단
  - 제안한 geometry-aware loss를 독립적인 주 기여로 사용하기 어려운 결과
  - Small lesion difficulty와 hierarchy consistency 자체는 확인
  - 기존 주제에서 기대한 성능 향상 근거는 부족
