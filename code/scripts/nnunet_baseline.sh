#!/usr/bin/env bash
# nnU-Net v2 baseline on OUR BraTS2021 split (runs entirely on the node, inside tmux).
# Trains 2d (matches our 2D study) then 3d_fullres, both nnUNetTrainer_500epochs, fold 0
# with custom splits_final.json = our train/val patients; our test patients held out in imagesTs.
set -ex
export DEBIAN_FRONTEND=noninteractive
source /venv/main/bin/activate 2>/dev/null || true
apt-get update -qq && apt-get install -y -qq tmux rsync pigz unzip > /dev/null 2>&1 || true
pip install -q --upgrade pip
pip uninstall -y -q torchaudio 2>/dev/null || true
pip install -q nnunetv2 pandas pyarrow monai
# torch pin guard (nnunetv2 deps may bump torch)
python - <<'EOF'
import torch
v = torch.__version__
assert torch.cuda.is_available(), "cuda unavailable"
print("torch", v, torch.cuda.get_device_name(0))
EOF

export nnUNet_raw=/workspace/nnunet/raw
export nnUNet_preprocessed=/workspace/nnunet/pre
export nnUNet_results=/workspace/nnunet/res
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results /workspace/data

# ---------- 1) BraTS raw download ----------
if [ ! -d /workspace/data/raw ] || [ -z "$(ls /workspace/data/raw 2>/dev/null)" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1 || true
  export PATH=$HOME/.local/bin:$PATH
  source /workspace/.secrets
  mkdir -p ~/.kaggle && printf '%s' "$KAGGLE_TOKEN" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
  cd /workspace/data
  uvx --python 3.12 --from kaggle kaggle datasets download \
      -d dschettler8845/brats-2021-task1 -f BraTS2021_Training_Data.tar -p /workspace/data
  [ -f BraTS2021_Training_Data.tar.zip ] && unzip -q BraTS2021_Training_Data.tar.zip && rm -f BraTS2021_Training_Data.tar.zip
  mkdir -p raw && tar -xf BraTS2021_Training_Data.tar -C raw && rm -f BraTS2021_Training_Data.tar
fi

# ---------- 2) convert to nnU-Net Dataset137 (region-based), honoring OUR split ----------
python /workspace/kast/scripts/nnunet_convert.py \
  --raw /workspace/data/raw --split-csv /workspace/kast/data_meta/split_seed42.csv \
  --out $nnUNet_raw/Dataset137_BraTS2021

# ---------- 3) plan & preprocess ----------
nnUNetv2_plan_and_preprocess -d 137 --verify_dataset_integrity -c 2d 3d_fullres

# ---------- 4) custom single-fold split ----------
python - <<'EOF'
import json, pandas as pd, os
df = pd.read_csv("/workspace/kast/data_meta/split_seed42.csv")
pats = df.drop_duplicates("patient_id")[["patient_id", "split"]]
tr = sorted(pats[pats.split == "train"].patient_id)
va = sorted(pats[pats.split == "val"].patient_id)
out = [{"train": tr, "val": va}]
p = os.path.join(os.environ["nnUNet_preprocessed"], "Dataset137_BraTS2021", "splits_final.json")
json.dump(out, open(p, "w"))
print(f"splits_final.json: train={len(tr)} val={len(va)}")
EOF

# ---------- 5) train ----------
export nnUNet_n_proc_DA=24
nnUNetv2_train 137 2d 0 -tr nnUNetTrainer_500epochs --npz 2>&1 | tee /workspace/nnunet_2d.log
nnUNetv2_train 137 3d_fullres 0 -tr nnUNetTrainer_500epochs --npz 2>&1 | tee /workspace/nnunet_3d.log

# ---------- 6) evaluate validation predictions with OUR metrics ----------
for CFG in 2d 3d_fullres; do
  python /workspace/kast/scripts/nnunet_eval.py \
    --pred $nnUNet_results/Dataset137_BraTS2021/nnUNetTrainer_500epochs__nnUNetPlans__$CFG/fold_0/validation \
    --gt $nnUNet_raw/Dataset137_BraTS2021/labelsTr \
    --out /workspace/runs/nnunet-$CFG-500ep
done
echo NNUNET_ALL_DONE
