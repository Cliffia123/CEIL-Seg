import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset.cub import CUBDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataset.imagenet import  ImageNetDataset
from utils.dataset.voc import VOCDataset
from utils.util import IMAGE_MEAN_VALUE, IMAGE_STD_VALUE


def data_loader(args):
    if args.arch == 'InceptionV3':
        args.resize_size = 299
        args.crop_size = 299

    transform_train = transforms.Compose([
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE)])

    transform_pca = A.Compose([
        A.Normalize(),
        A.Resize(256, 256),
        ToTensorV2()
    ])
    
    if args.VAL_CROP:
        transform_val = transforms.Compose([
            transforms.Resize((args.resize_size, args.resize_size)),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE),
        ])
    else:
        transform_val = transforms.Compose([
            transforms.Resize((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN_VALUE, IMAGE_STD_VALUE),
        ])

    if args.dataset == 'CUB':

        img_train = CUBDataset(
            root=args.data_root,
            datalist=os.path.join(args.data_list, 'train.txt'),
            transform=transform_train,
            is_train=True,
            is_pca=False
        )
        img_val = CUBDataset(
            root=args.data_root,
            datalist=os.path.join(args.data_list, 'test.txt'),
            transform=transform_val,
            is_train=False,
            is_pca=False
        )
        img_ddt = CUBDataset( #提取物主成分方向
            root=args.data_root,
            datalist=os.path.join(args.data_list, 'pca.txt'),
            transform=transform_pca,
            is_train=True,
            is_pca=True  
        )
        img_pca = CUBDataset( #提取训练数据集的伪标签
            root=os.path.join(args.data_root),
            datalist=os.path.join(args.data_list, 'train.txt'),
            transform=transform_pca,
            is_train=True,
            is_pca=True
        )
    elif args.dataset == 'ILSVRC':

        img_train = ImageNetDataset(
            root=os.path.join(args.data_root, 'train'),
            datalist=os.path.join(args.data_list, 'train.txt'),
            transform=transform_train,
            is_train=True,
            is_pca=False
        )

        img_pca = ImageNetDataset(
            root=os.path.join(args.data_root, 'train'),
            datalist=os.path.join(args.data_list, 'train.txt'),
            transform=transform_pca,
            is_train=True,
            is_pca=True
        )
        val_list = os.path.join(args.data_list, 'val.txt')

        img_val = ImageNetDataset(
            root=os.path.join(args.data_root, 'test'),
            datalist=val_list,
            transform=transform_val,
            is_train=False,
            is_pca=False
        )
        img_ddt = ImageNetDataset(
            root=os.path.join(args.data_root, 'train'),
            datalist=os.path.join(args.data_list, 'pca_3.txt'),
            transform=transform_val,
            is_train=True,
            is_pca=False
        )
    else:
        raise Exception("No matching dataset {}.".format(args.dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(img_train)
    else:
        train_sampler = None

    train_loader = DataLoader(img_train,
                              batch_size=args.batch_size,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=args.workers)

    train_loader_2 = DataLoader(img_pca,
                              batch_size=args.batch_size,
                              shuffle=False,
                              sampler=None,
                              num_workers=args.workers)

    val_loader = DataLoader(img_val,
                            batch_size=args.batch_size,
                            shuffle=False,
                            sampler=None,
                            num_workers=args.workers)

    ddt_loader = DataLoader(img_ddt,
                            batch_size=args.pca_batch_size,
                            shuffle=False,
                            sampler=None,
                            num_workers=args.workers)

    return train_loader, train_loader_2, val_loader, ddt_loader, train_sampler
