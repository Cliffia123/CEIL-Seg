

import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import xml.etree.ElementTree as ET

class ImageNetDataset(Dataset):
    def __init__(self, root=None, datalist=None, transform=None,is_train=True, is_pca=False):

        self.root = root
        self.datalist = datalist
        self.transform = transform
        self.is_train = is_train
        self.is_pca = is_pca
        self.transform_mask = A.Compose([
            A.Normalize(),
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
        ])
        self.transform_val = A.Compose([
            A.Normalize(),
            A.Resize(256, 256),
            A.CenterCrop(224, 224),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc'))

        self.cls2idx = np.load('datalist/ILSVRC/cls2idx.npy', allow_pickle=True).item()
        # self.top1top5 = np.load('', allow_pickle=True).item()

        image_class = []
        image_names= []
        image_labels_number = []
        image_id = []
        #n01494475/n01494475_6311.JPEG 4  [0,1]

        #0 ILSVRC2012_val_00000001.JPEG 65
        if is_train:
            with open(self.datalist) as f:
                for line in f:
                    info = line.strip().split()
                    image_class_, image_name = info[0].split("/")
                    image_class.append(image_class_)
                    image_names.append(image_name)
                    image_id.append(int(info[-1]))
                    image_labels_number.append(int(info[-1]))
        else:
            with open(self.datalist) as f:
                for line in f:
                    info = line.strip().split()
                    image_class_ = info[-1]
                    image_name_ = info[1]
                    image_class.append(image_class_)
                    image_names.append(image_name_)
                    image_id.append(int(info[0]))
                    image_labels_number.append(int(info[-1]))
        self.image_class = image_class
        self.image_names = image_names
        self.image_labels_number = image_labels_number
        self.image_id = image_id


    def __getitem__(self, idx):
        image_class = self.image_class[idx]
        image_name = self.image_names[idx]
        image_labels_number = self.image_labels_number[idx]
        image_id = self.image_id[idx]
        if self.is_train:
            if self.is_pca:
                image = cv2.imread(os.path.join(self.root, image_class, image_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.transform is not None:
                    image = self.transform(image=image)
                    image = image['image']
            else:
                #mask train
                image = cv2.imread(os.path.join(self.root, image_class, image_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                root = '/data0/caoxz/EIL/ilsvrc_eil_geometric/train_log/cxz/pca_maps'
                mask = cv2.imread(os.path.join(root, image_class, image_name)) / 255.0
                if self.transform is not None:
                    pair = self.transform_mask(image=image, mask=mask)
                    image,image_mask = pair['image'], pair['mask']
                return image, image_id, image_class, image_name, image_labels_number, image_mask
            #     image = Image.open(os.path.join(self.root, image_class, image_name)).convert('RGB')
            #     if self.transform is not None:
            #         image = self.transform(image)
            # return image, image_id, image_class, image_name, image_labels_number
        else: #返回 box
            # image = cv2.imread(os.path.join(self.root, image_name))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # tree = ET.parse('/data1/jiaming/data/imagenet/'+ '/val_xml/' + image_name.replace('.JPEG', '.xml'))
            # H, W, C = image.shape
            # bboxes = []
            # for obj in tree.findall('object'):
            #     cls = self.cls2idx[obj.find('name').text]
            #     if cls != image_labels_number:
            #         print(image_name, cls, image_labels_number)
            #         continue
            #     box = obj.find('bndbox')
            #     xmin, ymin, xmax, ymax = float(box.find('xmin').text), float(box.find('ymin').text), float(
            #         box.find('xmax').text), float(box.find('ymax').text)
            #     bboxes.append(
            #         [min(max(xmin, 0), W - 1), min(max(ymin, 0), H - 1), max(min(xmax, W - 1), 0),
            #          max(min(ymax, H - 1), 0),
            #          image_labels_number])
            #     pairs = self.transform_val(image=image, bboxes=bboxes)
            #     bboxes_ = pairs['bboxes']
            #     bboxes_ = [[box[0].item(), box[1].item(), box[2].item(), box[3].item()] for box in bboxes_]
            #     # print(bboxes_)
            #     if( [box[1] >= box[3]] for box in bboxes_):
            #         continue
            #
            # return pairs['image'], pairs['bboxes'], image_id, image_class, image_name, image_labels_number
            image = Image.open(os.path.join(self.root, image_name)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image, image_id, image_class, image_name, image_labels_number


    def __len__(self):
        return len(self.image_class)
