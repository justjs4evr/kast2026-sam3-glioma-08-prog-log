# Original hierarchy/boundary study

- 대상: Doyoon Choi 학생 연구노트 후속 정리
- 작성 기준일: 2026-08-14
- 범위: SAM3 기반 BraTS 2021 glioma sub-region segmentation의 hierarchy 및 boundary 제약 실험
- 포함 시점: 초기 가설 설정부터 기대한 성능 향상이 확인되지 않은 단계까지
- 평가 범위: validation 187명

## 읽는 순서

1. `docs/01_RESEARCH_NOTE_CONTINUATION.md`
   - 기존 연구노트 뒤에 이어서 사용할 진행 기록
2. `docs/02_METHODS_AND_RUNS.md`
   - 데이터, 모델, loss, metric, run 설정, 실행 방법
3. `docs/03_RESULTS.md`
   - 결과표, 해석, 실험상 한계
4. `results/`
   - 문서 수치의 근거가 되는 config, summary json, patient parquet
5. `code/`
   - 당시 실험에 사용한 코드의 관련 부분

## 폴더 구성

```text
code/kast2/       SAM3 wrapper, dataset, loss, train/eval, metric
code/scripts/     BraTS 전처리, hierarchy 후처리, nnU-Net 변환 및 평가
data_meta/        원본 저장소에서 복사한 환자 단위 split
docs/             연구노트용 진행 기록, 방법, 결과
results/          validation 결과 원자료
```

## 코드 범위

- `code/` 아래 파일: 현재 저장소의 실험 파일을 수정 없이 복사
- 일부 파일에 본 실험 이후 추가된 기능 포함
  - Paraphrase prompt
  - KiTS23 dataset option
  - 여러 prompt family의 cross-evaluation
- 본 문서에서 다루는 실험에 사용한 범위
  - BraTS 2021
  - Short prompt
  - DiceCE, SDF, boundary, containment loss
  - Hierarchy post-processing
- 위 범위를 벗어나는 코드 경로: 본 결과 해석 및 재현에서 미사용

## 확인된 결론

- Encoder LoRA 기본 Dice3D 세 seed 평균: WT 0.882, TC 0.861, ET 0.806
  - Seed 42 단일 run: 0.877 / 0.859 / 0.802
- SDF-weighted 및 Kervadec boundary loss 적용 후 의미 있는 Dice·HD95·NSD 향상 없음
- Containment loss: hierarchy violation 감소, Dice 개선 없음
- Hard/soft intersection: hierarchy violation 0%로 제거, Dice 손실 없음
- Small lesion Dice2D 약 0.58–0.63, large lesion 약 0.90–0.94
- nnU-Net 2D/3D: SAM3 기반 모델보다 높은 validation 성능
- 해석: 모델 자체의 분할 실패가 아니라, 추가한 geometry-aware objective의 기대 효과 미확인

## 제외 자료

- Prompt paraphrase 및 prompt robustness 분석
- Test set 결과
- KiTS23, VoxTell 결과
- Checkpoint와 prediction mask
- 내부 작업 기록 및 세션 로그
- 인증서, key, token 등 credential

## 결과 파일 규칙

- `summary_val_short.json`: run 단위 평균 지표와 size-tier 결과
- `patients_val_short.parquet`: 환자별 3D Dice, HD95, NSD, hierarchy violation
- `config.json`: seed, GPU, batch size, learning rate 등 실행 인자
- `postproc_val_short.*`: hierarchy intersection 전후 결과

모든 수치는 `-r2` run만 사용. `-r2`가 없는 구 코드 산출물은 미포함
