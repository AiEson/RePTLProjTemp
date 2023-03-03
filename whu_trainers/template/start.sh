#!/bin/bash
#template start.sh

python train.py \
  --dataset_dir ~/codes/datasets/WHU \
  --precision 16 \
  --img_size 512 \
  --num_workers 12 \
  --project BuildingSeg \
  --name {{train_name}} \
  --encoder_name {{encoder_name}} \
  --seed 2004 \
  --epochs 72 \
  --batch_size 16 \
  --acc_batch 1 \
  --gpus -1 \
  --use_swa True \
  --optim_name adamw \
  --sche_name reducelr \
  --dataset WHU \
