#!/bin/bash

python train.py \
  --dataset_dir ~/codes/datasets/buildingSegDataset \
  --precision 16 \
  --img_size 512 \
  --num_workers 8 \
  --project BuildingSeg \
  --name DeepLabv3-ResNet34 \
  --encoder_name resnet34 \
  --seed 2004 \
  --epochs 64 \
  --batch_size 16 \
  --acc_batch 1 \
  --gpus 3 \
  --use_swa True \
  --optim_name adamw \
  --sche_name coswarmrestart \

restart
