# Collaboratively Erasing Integrated Learning with Segmentation for Weakly Supervised Object Localization
PyTorch implementation of “Collaboratively Erasing Integrated Learning with Segmentation for Weakly Supervised Object Localization”.

# Prerequisite
First, you need to clone the CEIL-Seg repository from GitHub. Open your terminal and run the following command:
```
git clone https://github.com/Cliffia123/CEIL-Seg.git
cd CEIL-Seg
```
We recommend setting up a conda environment for the project:
```
conda create --name=ceil_seg python=3.7
conda activate ceil_seg
pip install -r requirements.txt
```
# Running

## Datasets
- CUB-200-2011 ([https://data.caltech.edu/](https://data.caltech.edu/))
- ILCVRC ([https://image-net.org/download.php](https://image-net.org/download.php))

## Checkpoints
Download the [checkpints](https://drive.google.com/drive/folders/1-wGJ-EW6KGy9u3tSVqseui9nKFVkMN2G?usp=drive_link) of CEIL_seg for CUB and ILSVRC and put them in the ``checkpoints'' folder (only used during evaluation).

**Eval the model using Top-1, Top5 and GT-known metrics**
```
# For CUB or ILSVRC evaluation
cd cam-seg or imagenet-seg
python test.py
```
All the evaluation files in trainditional way are can be found in [Google Drive](https://drive.google.com/drive/folders/1qzdcNpU6V8Y8xq-XRDPhudoSpgOMVoQb?usp=drive_link).

**Eval the model using accroding to the method in ["Evaluating Weakly Supervised Object Localization Methods Right"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.pdf)**
```
# for CUB or ILSVRC evaluation using MaxBoxAccV2
cd cam-seg-maxboxaccv2 or imagenet-seg-maxboxaccv2
python eval.py
```
All the evaluation files in MaxBoxAccv2 way are can be found in [Google Drive](https://drive.google.com/drive/folders/1fdggPJXqBgzxpH_-UjUc2ofVf476d8hK?usp=drive_link).

## Training

### Procuding CEIL Map
*Using the following scripts to train a model*
```
#!/bin/bash
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
--multiprocessing-distributed  --world-size 1 \
--workers 32 --arch ${arch}  --name ${name} --dataset ${dataset} \
--data-root ${data_root} --pretrained True --batch-size ${batch} \
--epochs ${epoch} --lr ${lr} --LR-decay ${decay} --wd ${wd} --nest True \
--erase-thr 0.6 --acol-cls False --VAL-CROP True --evaluate False \
--cam-thr 0.45 --loc True --mode ${mode} \
--resume CEIL/cub/train_log/cxz/model_cub.pth.tar
```
To traing a model for ILSVRC, you can change the above contents like dataset to  `ILSVRC', data_root="datasets/ImageNet" accrording the your settings. 

### Training a segmentation model with CEIL Map
- Mkdir `CUB' or  'ImageNet' dir and put the CEIL Map into cub-seg or imagenet-seg.
- Change the configuration in 'triain.py' file, such as the root of dataset and CEIL Map.
- Train with following scripts.
```
cd cub-seg or imagenet-seg
python train.py
```



