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
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from torch.utils.data import Dataset, DataLoader, dataset
from model import Model
from utils import IoU, gaussian
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

        self.samples  = []
        self.cls2idx  = np.load('../dataset/ImageNet2012/cls2idx.npy', allow_pickle=True).item()
        self.top1top5 = np.load(args.top1top5, allow_pickle=True).item()

        with open(args.list, 'r') as lines:
            for line in lines:
                id_, name, label = line.strip().split(' ')
                self.samples.append([id_, name, int(label)])
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
        id_, name, label = self.samples[idx]
        image  = cv2.imread(self.args.datapath+'val/ILSVRC2012_img_val/'+name)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        H,W,C  = image.shape
        box = self.boxes[int(id_)]
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
            cnt, cnt_top1, cnt_top5, cnt_loc, thresh = 0, 0, 0, 0, 0.85
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
                project_map = torch.sigmoid(pred)

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
                xmin, ymin, xmax, ymax = [item.item() for item in np.array(bboxes[0])]

                if IoU([x, y, x+w, y+h], [xmin, ymin, xmax, ymax] )>0.8:

                    cam_ = project_map.cpu().numpy().transpose(0, 2, 3, 1)

                    image_ = image.clone().detach().cpu() * image_mean + image_std
                    blend_tensor = torch.empty_like(image_)
                    image_ = image_.numpy().transpose(0, 2, 3, 1)
                    image_ = image_[:, :, :, ::-1] * 255
                    
                    blend_bbox = self.generate_map(image_[0], cam_[0], [xmin, ymin, xmax, ymax], [x, y, w, h], thresh)

                    blend_bbox = blend_bbox[:, :, ::-1] / 255.
                    blend_bbox = blend_bbox.transpose(2, 0, 1)
                    blend_tensor = torch.tensor(blend_bbox)
                    save_images('results', name[0], blend_tensor)

            print('thr=%5f count=%d | top1=%.5f | top5=%.5f | GT-Known=%.5f'%(thresh, cnt, cnt_top1/cnt, cnt_top5/cnt, cnt_loc/cnt))
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

def save_images(saving_folder, file_name, blend_tensor):
    if not os.path.isdir(saving_folder):
        os.makedirs(saving_folder)
    print(saving_folder, file_name)
    saving_path = os.path.join(saving_folder, file_name)
    vutils.save_image(blend_tensor, saving_path)         

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='/GPUFS/nsccgz_ywang_zfd/ImageNet/train/ILSVRC2012_img_train/data/')
    parser.add_argument('--top1top5'    ,default='../imagenet-cls/top1top5.npy')
    parser.add_argument('--snapshot'    ,default='./out/model-6000')
    parser.add_argument('--mode'        ,default='val')
    parser.add_argument('--list'        ,default='../dataset/ImageNet2012/val.txt')
    parser.add_argument('--batch_size'  ,default=1)
    parser.add_argument('--num_workers' ,default=1)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    # models = os.listdir('./out/')
    # for model in models:
    #     print("===========Model===========", model)
    #     args.snapshot = os.path.join('./out/', model)
    t = Test(Data, Model, args)
    t.box_precise()