import os
import random
import numpy as np
from numpy import *
import warnings
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from torchvision.utils import make_grid

import network as models
from utils.util_args import get_args
from utils.util_cam import *
from utils.util_loader import data_loader
from utils.util import \
    accuracy, adjust_learning_rate, \
    save_checkpoint, load_model, AverageMeter, IMAGE_MEAN_VALUE, IMAGE_STD_VALUE, calculate_IOU, draw_bbox, save_images, \
    save_erased_images
import cv2
# from utils.ddt.generate_box_imagenet import *
# from network.vgg16_acol import _vgg, _vgg_val
from utils.util import *
from collections import defaultdict
from network import vgg16_acol

import pickle
import json
from network.resnet import ResNet
torch.set_num_threads(4)
best_acc1 = 0


def main():
    args = get_args()
    print(args)
    # 可视化训练日志存储
    args.log_folder = os.path.join('./train_log', args.name)

    # ddt存放路径
    args.vis_path = '/data0/caoxz/ACoL_EIL/vis'

    # ddt训练图片存储路径
    args.erased_folder = '/data0/caoxz/ACoL_EIL/ddt_train'

    if not os.path.join(args.log_folder):
        os.makedirs(args.log_folder)

    # 配置：seed, gpu, dist_url, distribute,
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    global writer
    # 检测gpu等配置
    # if args.gpu == 0 and not args.evaluate:
    writer = SummaryWriter(logdir=args.log_folder)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.dataset == "CUB":
        num_classes = 200
    elif args.dataset == "ILSVRC":
        num_classes = 1000
    else:
        raise Exception("No dataset named {}".format(args.dataset))

    # Define Model 模型架构，基础是vgg16

    # model = models.__dict__[args.arch](pretrained=args.pretrained,
    #                                    num_classes=num_classes,
    #                                    drop_thr=args.erase_thr,
    #                                    )
    # model = ResNet(num_classes=num_classes)
    model = vgg16_acol._vgg(
        pretrained=True,
        progress=True,
        num_classes=num_classes,
    )
    param_features = []
    param_classifier = []

    # Give different learning rate
    for name, parameter in model.named_parameters():
        if 'fc.' in name or '.classifier' in name:
            # print("name added to cls part",name)
            param_classifier.append(parameter)
        else:
            # print("name added to feature part",name)
            param_features.append(parameter)
    optimizer = torch.optim.SGD([
        {'params': param_features, 'lr': args.lr},
        {'params': param_classifier, 'lr': args.lr * args.lr_ratio}],
        momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nest)

    # Change the last fully connected layers.
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint 从最新的检查点开始恢复
    if args.resume:
        model, optimizer = load_model(model, optimizer, args)

    # Loading training, validation, ddt dataset
    cudnn.benchmark = True
    train_loader, train_loader_2, val_loader, ddt_loader, train_sampler = data_loader(args)

    # 在验证集上测试模型  直接用vgg16预训练好的网络测试
    if args.evaluate:
        model, _ = load_model(model, optimizer, args)
        # model = models.__dict__[args.arch](pretrained=args.pretrained,
        #                                    num_classes=num_classes,
        #                                    drop_thr=args.erase_thr,
        #                                    )
        # args.mode = 'eva'
        # first_compo = defaultdict(dict)
        # # evaluate_cam(val_loader, model, 0, args, first_compo) #produce cam in train_loader_2
        # evaluate_loc(val_loader, model, 0, args, first_compo)# test in val_loader
        # return
        print("============> 5. 提取第一主成分方向 <============")
        args.mode = 'extra_compo'
        first_compo = extra_compo(ddt_loader, model, 0, args)

        print("============> 6. 测试 <============")
        args.mode = 'eva_pca'
        # evaluate_loc(val_loader, model, 0, args, first_compo)
        extra_compo_posi(train_loader_2, model, first_compo,0,args)
        return

    # pca_maps = defaultdict(dict)
    # pca_maps_ori = defaultdict(dict)
    # with open(os.path.join(args.json_root, "pca_maps_CUB_ori.json"), 'r') as load_f:
    #         load_dict = json.load(load_f)
    #         for key in load_dict.keys():
    #             pca_maps_ori[key] = torch.Tensor(load_dict[key])
    # del load_dict

    # 初始化训练
    print("============> 1. 开始初始化训练 <============")
    best_acc1 = 0
    # args.mode = 'train'
    args.mode = 'train_with_erasing'
    best_acc1 = train_mode(model, train_sampler, optimizer, train_loader, val_loader, ngpus_per_node, best_acc1, args)
    # best_acc1 = train_mode(model, train_sampler, optimizer, train_loader, val_loader, ngpus_per_node, pca_maps,best_acc1, args)

    # epoch_temp = args.epochs
    # for i in range(5):
    #     # #提取主成分方向
    #     print("============>" + str((i + 1) * 2) + ". 提取第一主成分方向 <============")
    #     args.mode = 'extra_compo'
    #     first_compo = extra_compo(ddt_loader, model, 0, args)

    #     # if not os.path.isdir(args.json_root):
    #     #     os.makedirs(args.json_root)
    #     # with open(os.path.join(args.json_root,'first_compo.json'), "w") as f:
    #     #     json.dump(first_compo, f)
    #     print(args.pca_maps + "文件写入完成...")

    #     print("============>" + str((i + 1) * 3) + ". 提取擦除位置 <============") #保存已经被擦除后的图像
    #     args.mode = 'extra_pca'
    #     pca_maps = extra_compo_posi(train_loader_2, model, first_compo, 0, args)

        # # if not os.path.isdir(args.json_root):
        # #     os.makedirs(args.json_root)
        # # with open(os.path.join(args.json_root, args.pca_maps), "w") as f:
        # #     json.dump(pca_maps, f)
        # # print(args.pca_maps + "文件写入完成...")

        # print(os.path.join(args.json_root, args.pca_maps) + "文件加载完成...")

        # # 擦除pca区域并且再训练
        # print("============>" + str((i + 1) * 4) + ". 开始擦除学习 <============") #加载擦除后的图像
    #     args.mode = 'train_with_erasing'
    #     args.start_epoch = args.epochs
    #     args.epochs = args.epochs + epoch_temp
    #     best_acc2 = train_mode(model, train_sampler, optimizer, train_loader, val_loader, ngpus_per_node, best_acc1, args)
    # if not os.path.isdir(args.json_root):
    #     os.makedirs(args.json_root)
    # with open(os.path.join(args.json_root, args.first_compo), "w") as f:
    #     json.dump(first_compo, f)
    # print("文件写入完成...")


def train_mode(model, train_sampler, optimizer, train_loader, val_loader, ngpus_per_node, best_acc1, args):
    temp_mode = args.mode
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        args.mode = temp_mode
        train_acc1, train_loss = train(train_loader, model, optimizer, epoch, args)

        # 默认loc为False
        args.mode = 'eva'
        val_acc1, val_loss = validate(val_loader, model, epoch, args)
        # val_acc1, val_acc5, top1_loc, top5_loc, gt_loc, val_loss = evaluate_loc(val_loader, model, epoch, args,
        #                                                                         fea ture_mask_dict)

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        if args.gpu == 0:
            print("Until %d epochs, Best Acc@1 %.3f" % (epoch + 1, best_acc1))

        # remember best acc@1 and save checkpoint
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.log_folder)
    return best_acc1


# 训练流程开始
def train(train_loader, model, optimizer, epoch, args):
    # AverageMeter for Performance  平均性能表
    losses = AverageMeter('Loss', ':.4e')
    loss_cls = AverageMeter('Loss', ':.4e')
    loss_center = AverageMeter('Loss', ':.4e')
    loss_cams = AverageMeter('Loss', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model.train()
    train_t = tqdm(train_loader)  # 进度条██████████

    for i, (images, mage_id, image_class, image_name, target) in enumerate(train_t):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        mask = torch.zeros(len(image_name), 1, 7, 7)

        mask = F.interpolate(mask, (14, 14), mode='bilinear', align_corners=True)[:, 0, :, :].unsqueeze(1)

        output = model(images, mode=args.mode, mask=mask, target=target)

        if args.distributed:
            loss = model.module.get_loss(target=target, mask= mask, mode=args.mode)
        else:
            loss = model.get_loss(target=target, mask=mask, mode=args.mode)

        acc1, acc5 = accuracy_pca(output, target, topk=(1, 5))

        if args.mode == 'train':

            losses.update(loss[0].data.item(), images.size(0))
            loss_cls.update(loss[1].data.item(), images.size(0))
            loss_center.update(loss[2].data.item(), images.size(0))

        else:
            losses.update(loss[0].data.item(), images.size(0))
            loss_cls.update(loss[1].data.item(), images.size(0))
            loss_center.update(loss[2].data.item(), images.size(0))
            loss_cams.update(loss[2].data.item(), images.size(0))

        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        description = "[T:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
            format(epoch, args.epochs, top1.avg, top5.avg, losses.avg)
        train_t.set_description(desc=description)
        optimizer.zero_grad()

        loss[0].backward()
        optimizer.step()

    return top1.avg, losses.avg


def validate(val_loader, model, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    feature_list_mask_dict = defaultdict(dict)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        val_t = tqdm(val_loader)
        for i, (images, image_ids, image_class, image_name, target) in enumerate(val_t):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.long().cuda(args.gpu, non_blocking=True)
            mask = torch.zeros(len(image_name), 1, 7, 7)

            # Compute output
            output = model(images, mode=args.mode, target=target)

            if args.distributed:
                loss = model.module.get_loss(target=target, mask=mask, mode=args.mode)
            else:
                loss = model.get_loss(target=target, mask=mask, mode=args.mode)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.data.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
                format(epoch, args.epochs, top1.avg, top5.avg, losses.avg)
            val_t.set_description(desc=description)

    return top1.avg, losses.avg


def extra_compo(val_loader, model, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    first_compo_list = defaultdict(dict)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        val_t = tqdm(val_loader)
        for i, (images, image_ids, image_class, image_name, target) in enumerate(val_t):
            # first_compo_list = defaultdict(dict)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            pca_tensor = torch.zeros(len(image_name), 1, 7, 7)

            output = model(images, compo=dict(first_compo_list), mode=args.mode, target=target)

            target_compo = target.cpu().numpy()

            first_compo = get_ddt_compo(model=model, args=args)

            first_compo_list[str(target_compo[0])] = first_compo

            target = target.long().cuda(args.gpu, non_blocking=True)

            if args.distributed:
                loss = model.module.get_class_loss(target)
            else:
                loss = model.get_class_loss(target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
                format(epoch, args.epochs, top1.avg, top5.avg, losses.avg)
            val_t.set_description(desc=description)

    return dict(first_compo_list)


def extra_compo_posi(train_loader, model, class_first_compo_list, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    feature_list_mask_dict = defaultdict(dict)


    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        val_t = tqdm(train_loader)
        for i, (images, image_ids, image_class, image_name, target) in enumerate(val_t):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            output = model(images, compo=dict(class_first_compo_list), mode=args.mode, target=target)

            feature_list_mask = get_cam_ddt_list(model=model, args=args)
            feature_list_mask = feature_list_mask.cpu().tolist()
            for j in range(len(image_name)):
                feature_list_mask_dict[image_name[j]] = feature_list_mask[j]

            target = target.long().cuda(args.gpu, non_blocking=True)
            if args.distributed:
                loss = model.module.get_class_loss(target)
            else:
                loss = model.get_class_loss(target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
                format(epoch, args.epochs, top1.avg, top5.avg, losses.avg)
            val_t.set_description(desc=description)

            pca_map_list = get_cam_ddt_list(model=model, args=args)
            pca_ = pca_map_list.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)

            for j in range(images.size(0)):
                pca = generate_pca(pca_[j])
                saving_folder = os.path.join(args.log_folder, 'pca_maps_cub', image_class[j])
                if not os.path.isdir(saving_folder):
                    os.makedirs(saving_folder)
                cv2.imwrite(os.path.join(saving_folder, image_name[j]), pca)

    return dict(feature_list_mask_dict)

def evaluate_cam(val_loader, model, epoch, args, first_compo):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    feature_list_mask_dict = defaultdict(dict)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        val_t = tqdm(val_loader)
        for i, (images, image_ids, image_class, image_name, target) in enumerate(val_t):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.long().cuda(args.gpu, non_blocking=True)

            output = model(images, compo=first_compo, mode=args.mode)

            if args.distributed:
                loss = model.module.get_class_loss(target)
            else:
                loss = model.get_class_loss(target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
                format(epoch, args.epochs, top1.avg, top5.avg, losses.avg)
            val_t.set_description(desc=description)

            cam_list = get_cam_ddt_list(model=model, args=args)
            cam_ = cam_list.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)

            for j in range(images.size(0)):
                cam = generate_pca(cam_[j])
                saving_folder = os.path.join(args.log_folder, 'cam', image_class[j])
                if not os.path.isdir(saving_folder):
                    os.makedirs(saving_folder)
                cv2.imwrite(os.path.join(saving_folder, image_name[j]), cam)

def evaluate_loc(val_loader, model, epoch, args, first_compo):
    losses = AverageMeter('Loss')

    top1_cls = AverageMeter('Acc@1')
    top5_cls = AverageMeter('Acc@5')

    # image
    gt_bbox = load_bbox(args)
    cnt = 0
    cnt_false_top1 = 0
    cnt_false_top5 = 0
    hit_known = 0
    hit_top1 = 0
    hit_top5 = 0
    iou_list = []
    image_mean = torch.reshape(torch.tensor(IMAGE_MEAN_VALUE), (1, 3, 1, 1))
    image_std = torch.reshape(torch.tensor(IMAGE_STD_VALUE), (1, 3, 1, 1))

    model.eval()
    with torch.no_grad():
        val_t = tqdm(val_loader)
        for i, (images, image_ids, image_class, image_name, target) in enumerate(val_t):

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.long().cuda(args.gpu, non_blocking=True)
            # if len(bboxes) == 0:
            #     # print(image_name)
            #     continue
            # bboxes = [[box[0].item(), box[1].item(), box[2].item(), box[3].item()] for box in bboxes]

            output = model(images, compo=first_compo, mode=args.mode,target=target)

            if args.distributed:
                loss = model.module.get_class_loss(target)
            else:
                loss = model.get_class_loss(target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # top1, top5 = top1top5

            losses.update(loss.item(), images.size(0))

            top1_cls.update(acc1[0], images.size(0))
            top5_cls.update(acc5[0], images.size(0))

            description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, Loss: {4:7.4f}, ". \
                format(epoch, args.epochs, top1_cls.avg, top5_cls.avg, losses.avg)
            val_t.set_description(desc=description)

            _, pred = output.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_1 = correct[:1].flatten(1).float().sum(dim=0)
            correct_5 = correct[:5].flatten(1).float().sum(dim=0)

            # Get cam 得到得到分类器A和B中特征最突出的热力图像
            # 选取cams图片
            cam_ddt_list = get_cam_ddt_list(model=model, args=args)
            image_ = images.clone().detach().cpu() * image_mean + image_std
            image_ = image_.numpy().transpose(0, 2, 3, 1)
            image_ = image_[:, :, :, ::-1] * 255
            cam_ = cam_ddt_list.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)

            for j in range(images.size(0)):

                # gray_map = generate_psedo_box( cam_[j])
                estimated_bbox, blend_bbox, heatmap = generate_bbox(image_[j],
                                                           cam_[j],
                                                           gt_bbox[image_name[j][:-5]][0],
                                                           args.cam_thr, args)
                iou = calculate_IOU(gt_bbox[image_name[j][:-5]], estimated_bbox)
                # saving_folder = os.path.join(args.log_folder, 'pca_val')
                # if not os.path.isdir(saving_folder):
                #     os.makedirs(saving_folder)
                # cv2.imwrite(os.path.join(saving_folder, image_name[j]), blend_bbox)
                cnt += 1
                if iou >= 0.5:
                    iou_list.append(iou)
                    hit_known += 1
                    if correct_5[j] > 0:
                        hit_top5 += 1
                        if correct_1[j] > 0:
                            hit_top1 += 1
                        elif correct_1[j] == 0:
                            cnt_false_top1 += 1
                    elif correct_5[j] == 0:
                        cnt_false_top1 += 1
                        cnt_false_top5 += 1
                else:
                    if correct_5[j] > 0:
                        if correct_1[j] == 0:
                            cnt_false_top1 += 1
                    elif correct_5[j] == 0:
                        cnt_false_top1 += 1
                        cnt_false_top5 += 1

                # save_images('results_total', 0, i, blend_tensor, args)

            description = "[V:{0:3d}/{1:3d}] Top1-cls: {2:6.2f}, Top5-cls: {3:6.2f}, " \
                          "" \
                          "+: {4:7.4f}, ". \
                format(epoch, args.epochs, top1_cls.avg, top5_cls.avg, losses.avg)
            val_t.set_description(desc=description)

        loc_gt = hit_known / cnt * 100
        loc_top1 = hit_top1 / cnt * 100
        loc_top5 = hit_top5 / cnt * 100
        cls_top1 = (1 - cnt_false_top1 / cnt) * 100
        cls_top5 = (1 - cnt_false_top5 / cnt) * 100

        if args.gpu == 0:
            print("Evaluation Result:\n"
                  "LOC GT:{0:6.2f} Top1: {1:6.2f} Top5: {2:6.2f}\n"
                  "CLS TOP1: {3:6.3f} Top5: {4:6.3f}".
                  format(loc_gt, loc_top1, loc_top5, cls_top1, cls_top5))
    return cls_top1, cls_top5, loc_top1, loc_top5, loc_gt, losses.avg


if __name__ == '__main__':
    main()
