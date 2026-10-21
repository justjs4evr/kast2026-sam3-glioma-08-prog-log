# Result artifact map

| 폴더 | 내용 |
|---|---|
| `p1-seg_head_ft-*` | Segmentation head-only PEFT |
| `p1-lora_enc+seg_head-*` | Encoder LoRA + segmentation head PEFT |
| `p2-lora_enc-short-dicece-*` | DiceCE 기준선, seeds 42/2026/67 |
| `p3-lora_enc-short-sdf-*` | SDF-weighted loss, seeds 42/2026/67 |
| `p3-lora_enc-short-boundary-*` | Kervadec boundary loss, seeds 42/2026/67 |
| `p3-lora_enc-short-containment-*` | Containment loss, seed 42 |
| `hierarchy_postproc/` | None/hard/soft 후처리 결과 |
| `nnunet-2d-500ep/` | nnU-Net 2D validation 결과 |
| `nnunet-3d-500ep/` | nnU-Net 3D full-resolution validation 결과 |

각 SAM3 run에 포함된 파일:

- `config.json`: 실행 인자 및 GPU
- `summary_val_short.json`: validation 평균 및 size-tier 지표
- `patients_val_short.parquet`: 환자별 3D 지표

제외 항목:

- `best.pt`, `last.pt`
- Slice별 prediction mask
- Test split 결과
- 다른 prompt family 평가 결과
