#!/bin/bash
set -euo pipefail

devices=(0 1 2 3 4 5)
DATASET="POINTCLOUD"
models=("DeepSets" "SetTransformer")
s1_sizes=(128 256 512)
num_devices=${#devices[@]}

VENV_PATH="/mnt/nas/soutrik/Monotone-Embedding/env/bin/activate"
FILE_NAME_SUFFIX="results_all_pointcloud_LATEST"

count=0
for MODEL in "${models[@]}"; do
  for s1_size in "${s1_sizes[@]}"; do
    device=${devices[$((count % num_devices))]}
    if [ "$MODEL" == "SetTransformer" ]; then
      echo "Launching model=$MODEL on DEVICE=$device for DATASET=$DATASET with S1 Size=$s1_size"
      (
        source "$VENV_PATH"
        # export CUDA_VISIBLE_DEVICES="$device"   # uncomment if needed
        python train.py \
          --model_type "$MODEL" \
          --DEVICE "$device" \
          --DATASET_NAME "$DATASET" \
          --file_name_suffix "$FILE_NAME_SUFFIX" \
          --d 20 \
          --s1_size "$s1_size"
      ) &
      count=$((count + 1))
      if (( count % num_devices == 0 )); then wait; fi
    else
      echo "Launching model=$MODEL on DEVICE=$device (no outer rho) for DATASET=$DATASET with S1 Size=$s1_size"
      (
        source "$VENV_PATH"
        # export CUDA_VISIBLE_DEVICES="$device"
        python train.py \
          --model_type "$MODEL" \
          --no_outer_rho \
          --DEVICE "$device" \
          --DATASET_NAME "$DATASET" \
          --file_name_suffix "$FILE_NAME_SUFFIX" \
          --d 20 \
          --s1_size "$s1_size"
      ) &
      count=$((count + 1))
      if (( count % num_devices == 0 )); then wait; fi

      device=${devices[$((count % num_devices))]}
      echo "Launching model=$MODEL on DEVICE=$device (with outer rho) for DATASET=$DATASET with S1 Size=$s1_size"
      (
        source "$VENV_PATH"
        # export CUDA_VISIBLE_DEVICES="$device"
        python train.py \
          --model_type "$MODEL" \
          --DEVICE "$device" \
          --DATASET_NAME "$DATASET" \
          --file_name_suffix "$FILE_NAME_SUFFIX" \
          --d 20 \
          --s1_size "$s1_size"
      ) &
      count=$((count + 1))
      if (( count % num_devices == 0 )); then wait; fi
    fi
  done
done

wait
echo "All runs complete."
