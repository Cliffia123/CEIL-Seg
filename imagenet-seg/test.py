#coding=utf-8

import sys
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
plt.ion()
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, dataset
from model import Model
from utils import IoU, gaussian
import albumentations as A
from albumentations.pytorch import ToTensorV2


class Data(Dataset):
    def __init__(self, args):
        self.args      = args
        self.transform = A.Compose([
            A.Normalize(),
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        self.samples  = []
        self.cls2idx  = np.load('../dataset/ImageNet2012/cls2idx.npy', allow_pickle=True).item()
        self.top1top5 = np.load(args.top1top5, allow_pickle=True).item()

        with open(args.list, 'r') as lines:
            for line in lines:
                id, name, label = line.strip().split(' ')
                self.samples.append([id, name, int(label)])
        self.boxes = self.load_box()   
        
    def load_box(dataset_path='../dataset/ImageNet2012/', resize_size=256, crop_size=224):
        origin_bbox = {}
        image_sizes = {}
        resized_bbox = {}

        with open('../dataset/ImageNet2012/bounding_boxes.txt', 'r') as lines:
            for each_line in lines:
                file_info = each_line.strip().split()
                image_id = int(file_info[0])
                # x, y, bbox_width, bbox_height =  map(float, file_info[1:])
                # origin_bbox[image_id] = [x, y, bbox_width, bbox_height]
                boxes =  map(float, file_info[1:])
                origin_bbox[image_id] = list(boxes)

        return origin_bbox

    def __getitem__(self, idx):
        id, name, label = self.samples[idx]
        image  = cv2.imread(self.args.datapath+'ILSVRC2012_img_val/'+name)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        H,W,C  = image.shape
        box = self.boxes[int(id)]
        num_boxes =len(box)//4
        bboxes = []
        for i in range(num_boxes):
            x, y, bbox_width, bbox_height = box[i:i+4]
            # print("box:", x, y, bbox_width, bbox_height)
            xmin, ymin, xmax, ymax = x, y, x+bbox_width, y+bbox_height

            xmin, ymin, xmax, ymax = min(max(xmin, 0), W-1), min(max(ymin, 0), H-1), max(min(xmax, W-1), 0), max(min(ymax, H-1),0)
            if ymin == ymax:
                ymax += 1   
            if xmin == xmax:
                xmax += 1     

            bboxes.append([xmin, ymin, xmax, ymax, label])
            # bboxes.append([xmin, ymin, xmax, ymax, label])

            pairs = self.transform(image=image, bboxes=bboxes)

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
            thresh_list = [0.82, 0.84, 0.86, 0.88]
            for thresh in thresh_list:
                cnt, cnt_top1, cnt_top5, cnt_loc = 0, 0, 0, 0
                for name, origin, image, bboxes, top1top5 in self.loader:
                    image = image.cuda().float()
                    top1, top5 = top1top5
                    if len(bboxes)==0:
                        print(name)
                        continue
                    label  = bboxes[0][-1].item()
                    bboxes = [[box[0].item(), box[1].item(), box[2].item(), box[3].item()] for box in bboxes]
                    pred   = self.model(image)
                    pred   = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=True)
                    pred   = torch.sigmoid(pred)[0, 0, :, :].cpu().numpy()

                    ## accuracy
                    mask     = np.uint8(pred>thresh)*255
                    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if len(contours)==0:
                        continue
                    contour  = max(contours, key=cv2.contourArea)
                    x,y,w,h  = cv2.boundingRect(contour)
                    if max([IoU([x, y, x+w, y+h],  box) for box in np.array(bboxes)])>0.5:
                        cnt_loc  += 1
                        cnt_top1 += 1 if top1 else 0
                        cnt_top5 += 1 if top5 else 0
                    cnt += 1
                    # if cnt%100 == 0:
                    #     print('count=%d | top1=%.5f | top5=%.5f | GT-Known=%.5f'%(cnt, cnt_top1/cnt, cnt_top5/cnt, cnt_loc/cnt))
                print('thr=%5f count=%d | top1=%.5f | top5=%.5f | GT-Known=%.5f'%(thresh, cnt, cnt_top1/cnt, cnt_top5/cnt, cnt_loc/cnt))

            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='/GPUFS/nsccgz_ywang_zfd/ImageNet/')
    parser.add_argument('--top1top5'    ,default='../imagenet-cls/top1top5.npy')
    parser.add_argument('--snapshot'    ,default='./out/model-2000')
    parser.add_argument('--mode'        ,default='val')
    parser.add_argument('--list'        ,default='../dataset/ImageNet2012/val.txt')
    parser.add_argument('--batch_size'  ,default=1)
    parser.add_argument('--num_workers' ,default=1)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    models = os.listdir('./out/')
    # for model in models:
    #     print("===========Model===========", model)
    #     args.snapshot = os.path.join('./out/', model)
    t = Test(Data, Model, args)
    t.box_precise()