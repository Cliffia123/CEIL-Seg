#!/bin/bash

# ACoL - VGG16 script

gpu=0,1
arch=vgg16_acol
name=cxz
dataset=CUB
data_root="datasets/CUB_200_2011/images"
decay=40
epoch=100
batch=128
wd=1e-4
lr=0.001
mode="train"

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
--multiprocessing-distributed \
--world-size 1 \
--workers 32 \
--arch ${arch} \
--name ${name} \
--dataset ${dataset} \
--data-root ${data_root} \
--pretrained True \
--batch-size ${batch} \
--epochs ${epoch} \
--lr ${lr} \
--LR-decay ${decay} \
--wd ${wd} \
--nest True \
--erase-thr 0.6 \
--acol-cls False \
--VAL-CROP True \
--evaluate False \
--cam-thr 0.45 \
--loc True \
--mode ${mode} \
--resume CEIL/cub/train_log/cxz/model_cub.pth.tar



