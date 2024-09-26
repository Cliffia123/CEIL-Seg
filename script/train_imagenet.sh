#!/bin/bash

# ACoL - VGG16 script
gpu=1,2
arch=vgg16_acol
name=cxz
dataset=ILSVRC
data_root="/data1/jiaming/data/imagenet/"
epoch=80
decay=40
batch=128
wd=1e-4
lr=0.001
mode="train"

CUDA_VISIBLE_DEVICES=${gpu} python main.py \
--multiprocessing-distributed \
--world-size 1 \
--workers 30 \
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
--label-folder False \
--cam-thr 0.3 \
--loc True \
--mode ${mode} \
--resume /data0/caoxz/EIL/ilsvrc_eil_geometric/train_log/cxz/model_eil_ori.pth.tar
