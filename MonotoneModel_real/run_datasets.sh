#!/bin/bash
set -euo pipefail

devices=(0 1 2 3 4 5 6)
datasets=("MSWEB" "MSNBC" "AMAZON_bedding" "AMAZON_feeding")
models=("DeepSets" "SetTransformer")
num_devices=${#devices[@]}

VENV_PATH="/mnt/nas/soutrik/Monotone-Embedding/env/bin/activate"
FILE_NAME_SUFFIX="results_all_datasets"

count=0
for DATASET in "${datasets[@]}"; do
  for MODEL in "${models[@]}"; do
    device=${devices[$((count % num_devices))]}
    if [ "$MODEL" == "SetTransformer" ]; then
      echo "Launching model=$MODEL on DEVICE=$device for DATASET=$DATASET"
      (
        source "$VENV_PATH"
        python train.py \
          --model_type "$MODEL" \
          --DEVICE "$device" \
          --DATASET_NAME "$DATASET" \
          --file_name_suffix "$FILE_NAME_SUFFIX"
      ) &
      count=$((count + 1))
      if (( count % num_devices == 0 )); then wait; fi
    else
      echo "Launching model=$MODEL on DEVICE=$device (no outer rho) for DATASET=$DATASET"
      (
        source "$VENV_PATH"
        python train.py \
          --model_type "$MODEL" \
          --no_outer_rho \
          --DEVICE "$device" \
          --DATASET_NAME "$DATASET" \
          --file_name_suffix "$FILE_NAME_SUFFIX"
      ) &
      count=$((count + 1))
      if (( count % num_devices == 0 )); then wait; fi

      device=${devices[$((count % num_devices))]}
      echo "Launching model=$MODEL on DEVICE=$device (with outer rho) for DATASET=$DATASET"
      (
        source "$VENV_PATH"
        python train.py \
          --model_type "$MODEL" \
          --DEVICE "$device" \
          --DATASET_NAME "$DATASET" \
          --file_name_suffix "$FILE_NAME_SUFFIX"
      ) &
      count=$((count + 1))
      if (( count % num_devices == 0 )); then wait; fi

      device=${devices[$((count % num_devices))]}
      echo "Launching model=$MODEL on DEVICE=$device (with monotone m2) for DATASET=$DATASET"
      (
        source "$VENV_PATH"
        python train.py \
          --model_type "$MODEL" \
          --monotone_m2 \
          --DEVICE "$device" \
          --DATASET_NAME "$DATASET" \
          --file_name_suffix "$FILE_NAME_SUFFIX"
      ) &
      count=$((count + 1))
      if (( count % num_devices == 0 )); then wait; fi
    fi
  done
done

wait
echo "All runs complete."
