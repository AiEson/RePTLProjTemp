#!/bin/bash
#template start.sh

python test.py \
  --dataset_dir ~/codes/datasets/Massachusetts_cropped512 \
  --precision 16 \
  --img_size 512 \
  --num_workers 8 \
  --project BuildingSeg \
  --name MASS_ResNeSt101e-GSoP-SCA-UPP \
  --encoder_name timm-resnest101e-meangsop-sca \
  --seed 2004 \
  --epochs 120 \
  --batch_size 8 \
  --acc_batch 2 \
  --gpus -1 \
  --use_swa True \
  --optim_name adamw \
  --sche_name reducelr \
  --dataset MASS \
  --lr 0.0005 \
  --best_ckpt_filename BinaryJaccardIndex=0.5560.ckpt \