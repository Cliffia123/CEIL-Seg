# Collaboratively Erasing Integrated Learning with Segmentation for Weakly Supervised Object Localization
PyTorch implementation of “Collaboratively Erasing Integrated Learning with Segmentation for Weakly Supervised Object Localization”.

# Prerequisite
First, clone the CEIL-Seg repository from GitHub by opening your terminal and running the following command:
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
- ILSVRC ([https://image-net.org/download.php](https://image-net.org/download.php))

## Checkpoints
Download the [checkpints](https://drive.google.com/drive/folders/1-wGJ-EW6KGy9u3tSVqseui9nKFVkMN2G?usp=drive_link) of CEIL-Seg for CUB and ILSVRC and place them in the checkpoints folder (used only during evaluation).

**Eval the model using Top-1, Top5 and GT-known metrics**
```
# For CUB or ILSVRC evaluation
cd cam-seg or imagenet-seg
python test.py
```
All traditional evaluation files can be found in [Google Drive](https://drive.google.com/drive/folders/1qzdcNpU6V8Y8xq-XRDPhudoSpgOMVoQb?usp=drive_link).

**Evaluate the model according to the method in ["Evaluating Weakly Supervised Object Localization Methods Right"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choe_Evaluating_Weakly_Supervised_Object_Localization_Methods_Right_CVPR_2020_paper.pdf)**
First, download the metadata and move it into the cub200-seg or ImageNet-Seg directories:
```
# for CUB or ILSVRC evaluation using MaxBoxAccV2
cd cam-seg-maxboxaccv2 or imagenet-seg-maxboxaccv2
python eval.py
```
All evaluation files for MaxBoxAccv2 can be found in  [Google Drive](https://drive.google.com/drive/folders/1fdggPJXqBgzxpH_-UjUc2ofVf476d8hK?usp=drive_link).

## Training

### Procuding CEIL Map
*Using the following script to train a model*
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
To train a model for ILSVRC, modify the dataset to ILSVRC and the data_root to datasets/ImageNet according to your settings.

### Training a segmentation model with CEIL Map
- Create a CUB or ImageNet directory and place the CEIL Map in cub-seg or imagenet-seg.
- Modify the configuration in the train.py file, such as the dataset root and CEIL Map.
- Train the model using the following script:

```
cd cub-seg or imagenet-seg
python train.py
```