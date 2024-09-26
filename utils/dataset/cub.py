import cv2
from operator import is_
import os
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CUBDataset(Dataset):
    def __init__(self, root=None, datalist=None, transform=None, is_train=True, is_pca=False):

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
        image_class = []
        image_names= []
        image_labels_number = []
        image_id = []
        #7 001.Black_footed_Albatross/Black_Footed_Albatross_0031_100.jpg 0
        with open(self.datalist) as f:
            for line in f:
                info = line.strip().split()
                image_class_, image_name = info[1].split("/")
                image_class.append(image_class_)
                image_names.append(image_name)
                image_id.append(int(info[0]))
                image_labels_number.append(int(info[2]))
        self.image_class = image_class
        self.image_names = image_names
        self.image_labels_number = image_labels_number
        self.image_id = image_id

    def __getitem__(self, idx):
        image_class = self.image_class[idx]
        image_name = self.image_names[idx]
        image_labels_number = self.image_labels_number[idx]
        image_id = self.image_id[idx]
        # image = Image.open(os.path.join(self.root, image_class, image_name)).convert('RGB')

        # if self.transform is not None:
        #     image = self.transform(image)
        #     print("coming--------")
        if self.is_train:   
            if self.is_pca:
                image = cv2.imread(os.path.join(self.root, image_class, image_name))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # if self.transform is not None:
                image = self.transform(image=image)
                image = image['image']
            else:
                image = Image.open(os.path.join(self.root, image_class, image_name)).convert('RGB')
                # if self.transform is not None:
                image = self.transform(image)

                # #--------mask----------
                # image = cv2.imread(os.path.join(self.root, image_class, image_name))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # root = '/data0/caoxz/EIL/ilsvrc_eil_geometric/train_log/cxz/pca_maps_cub'
                # mask = cv2.imread(os.path.join(root, image_class, image_name)) / 255.0
                # if self.transform is not None:
                #     pair = self.transform_mask(image=image, mask=mask)
                #     image,image_mask = pair['image'], pair['mask']
                # return image, image_id, image_class, image_name, image_labels_number, image_mask
        else:
            image = Image.open(os.path.join(self.root, image_class, image_name)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        return image, image_id, image_class, image_name, image_labels_number


    def __len__(self):
        return len(self.image_class)
