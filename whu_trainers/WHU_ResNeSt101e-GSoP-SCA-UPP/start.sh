#!/bin/bash
#template start.sh

python train.py \
  --dataset_dir ~/codes/datasets/WHU \
  --precision 16 \
  --img_size 512 \
  --num_workers 8 \
  --project BuildingSeg \
  --name WHU_ResNeSt101e-GSoP-SCA-UPP \
  --encoder_name timm-resnest101e-meangsop-sca \
  --seed 2004 \
  --steps 44500 \
  --batch_size 8 \
  --acc_batch 2 \
  --gpus -1 \
  --use_swa False \
  --optim_name adamw \
  --sche_name reducelr \
  --dataset WHU \
  --lr 0.0001 \
