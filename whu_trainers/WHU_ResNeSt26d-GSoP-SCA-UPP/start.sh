#!/bin/bash
#template start.sh

python train.py \
  --dataset_dir ~/codes/datasets/WHU \
  --precision 16 \
  --img_size 512 \
  --num_workers 8 \
  --project BuildingSeg \
  --name WHU_ResNeSt26d-GSoP-SCA-UPP \
  --encoder_name timm-resnest26d-meangsop-sca \
  --seed 2004 \
  --steps 44500 \
  --epochs 1000 \
  --batch_size 16 \
  --acc_batch 1 \
  --gpus -1 \
  --use_swa True \
  --optim_name adamw \
  --sche_name reducelr \
  --dataset WHU \
  --lr 0.0001 \
