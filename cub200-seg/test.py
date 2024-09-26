#coding=utf-8
import sys
import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

from model import Model
from utils import IoU
import albumentations as A
from albumentations.pytorch import ToTensorV2
IMAGE_MEAN_VALUE = [0.485, .456, .406]
IMAGE_STD_VALUE = [.229, .224, .225]

image_mean = torch.reshape(torch.tensor(IMAGE_MEAN_VALUE), (1, 3, 1, 1))
image_std = torch.reshape(torch.tensor(IMAGE_STD_VALUE), (1, 3, 1, 1))
class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))
        
        self.top1top5 = np.load(args.top1top5, allow_pickle=True).item()
        self.samples  = []
        with open(args.label, 'r') as lines, open(args.list, 'r') as lines2:
            for line1, line2 in zip(lines, lines2):
                name, label            = line1.split(',')
                name, xmin, ymin, xmax, ymax   = line2.split(',')
                # xmin, ymin, xmax, ymax = box.split(' ')
                self.samples.append([name, [[float(xmin), float(ymin), float(xmax), float(ymax), int(label)]]])

    def __getitem__(self, idx):
        name, bboxes = self.samples[idx]
        image = cv2.imread(self.args.datapath+'/images/'+name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H,W,_ = image.shape
        bboxes_clip = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, label = bbox
            bboxes_clip.append([max(xmin, 0), max(ymin, 0), min(xmax, W-1), min(ymax, H-1), label])
        pairs = self.transform(image=image, bboxes=bboxes_clip)
        return name, image, pairs['image'], pairs['bboxes'], self.top1top5[name]

    def __len__(self):
        return len(self.samples)


class Test(object):
    def __init__(self, Data, Model, args):
        self.args   = args
        ## dataset
        self.data   = Data(args)
        self.loader = DataLoader(self.data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        ## network
        self.model  = Model(args)
        self.model.train(False)
        self.model.cuda()
    
    def box_precise(self):
        with torch.no_grad():
            thresh_list = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            for thresh in thresh_list:
                cnt, cnt_top1, cnt_top5, cnt_loc, cnt_loc_30, cnt_loc_70 = 0, 0, 0, 0, 0, 0
                for name, origin, image, bboxes, (top1, top5) in self.loader:
                    xmin, ymin, xmax, ymax, label = [item.item() for item in bboxes[0]]
                    image      = image.cuda().float()

                    ## forward
                    pred  = self.model(image)
                    pred  = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=True)
                    project_map = torch.sigmoid(pred)
                    pred  = torch.sigmoid(pred)[0, 0, :, :].cpu().numpy()

                    ## accuracy
                    mask     = np.uint8(pred>thresh)*255
                    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if len(contours)==0:
                        continue
                    contour  = max(contours, key=cv2.contourArea)
                    x,y,w,h  = cv2.boundingRect(contour)
                    
                    if IoU([x, y, x+w, y+h],  [xmin, ymin, xmax, ymax])>0.5:
                        cnt_loc  += 1
                        cnt_top1 += 1 if top1 else 0
                        cnt_top5 += 1 if top5 else 0

                    if IoU([x, y, x+w, y+h],  [xmin, ymin, xmax, ymax])>0.3:
                        cnt_loc_30  += 1
                    
                    if IoU([x, y, x+w, y+h],  [xmin, ymin, xmax, ymax])>0.7:
                        cnt_loc_70  += 1
                    
                    cnt += 1
                print('thr=%5f, count=%d | top1=%.5f | top5=%.5f | GT-Known_50=%.5f'%(thresh, cnt, cnt_top1/cnt, cnt_top5/cnt, cnt_loc/cnt))
                print('thr=%5f, count=%d | IOU_30=%.5f,IOU_50=%.5f,IOU_70=%.5f, IOU_mean=%.5f'%(thresh, cnt, cnt_loc_30/cnt,cnt_loc/cnt, cnt_loc_70/cnt, (cnt_loc_30+cnt_loc+cnt_loc_30)/(3*cnt)))
                


    def generate_map(self, image, cam,  gt, pred_box, thr):

        image_height, image_width, _ = image.shape
        cam = cv2.resize(cam, (image_height, image_width),
                     interpolation=cv2.INTER_CUBIC)
        heatmap = intensity_to_rgb(cam, normalize=True).astype('uint8')
        heatmap_BGR = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)            
        
        blend = image * 0.5 + heatmap_BGR * 0.5

        x, y, w, h = pred_box
        cv2.rectangle(blend, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(blend, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 0, 255), 2)
        return blend


def intensity_to_rgb(intensity, cmap='cubehelix', normalize=False):
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return intensity.astype('float32') * 255.0

def save_images(log_folder, folder_name, file_name, blend_tensor):
    saving_folder = os.path.join(log_folder, folder_name)
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    saving_path = os.path.join(saving_folder, file_name)
    vutils.save_image(blend_tensor, saving_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='/GPUFS/nsccgz_ywang_zfd/caoxz/data/CUB_200_2011')
    parser.add_argument('--top1top5'    ,default='../cub200-cls/top1top5.npy')
    parser.add_argument('--snapshot'    ,default='./out/model-8')
    parser.add_argument('--mode'        ,default='val')
    parser.add_argument('--label'        ,default='/GPUFS/nsccgz_ywang_zfd/caoxz/SPOL/metadata/CUB/test/class_labels.txt')
    parser.add_argument('--list'        ,default='/GPUFS/nsccgz_ywang_zfd/caoxz/SPOL/metadata/CUB/test/localization.txt')
    parser.add_argument('--clsnum'      ,default=200)
    parser.add_argument('--batch_size'  ,default=1)
    parser.add_argument('--num_workers' ,default=8)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    # models = os.listdir('./out_0.2/')
    # for model in models:
    # print("===========Model===========", model)
    args.snapshot = os.path.join('./out_0.2_83.53/', 'model-1')
    t = Test(Data, Model, args)
    t.box_precise()