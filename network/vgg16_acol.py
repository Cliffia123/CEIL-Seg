import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import OrderedDict
from torch.utils.model_zoo import load_url
import torch.utils.model_zoo as model_zoo

import torchvision.models as models
import numpy as np
from utils.ddt.ddt_func import *
from torch.autograd import Variable
from utils.ddt.pca_project import *
from network.vgg16_cel import *
from network.vgg16_eil import *

__all__ = [
    'VGG', 'vgg16_acol',
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}
layer_mapping_vgg = OrderedDict([('features.0.weight', 'conv1_1.weight'), ('features.0.bias', 'conv1_1.bias'), ('features.2.weight', 'conv1_2.weight'), ('features.2.bias', 'conv1_2.bias'), ('features.5.weight', 'conv2_1.weight'), ('features.5.bias', 'conv2_1.bias'), ('features.7.weight', 'conv2_2.weight'), ('features.7.bias', 'conv2_2.bias'), ('features.10.weight', 'conv3_1.weight'), ('features.10.bias', 'conv3_1.bias'), ('features.12.weight', 'conv3_2.weight'), ('features.12.bias', 'conv3_2.bias'), ('features.14.weight', 'conv3_3.weight'), (
    'features.14.bias', 'conv3_3.bias'), ('features.17.weight', 'conv4_1.weight'), ('features.17.bias', 'conv4_1.bias'), ('features.19.weight', 'conv4_2.weight'), ('features.19.bias', 'conv4_2.bias'), ('features.21.weight', 'conv4_3.weight'), ('features.21.bias', 'conv4_3.bias'), ('features.24.weight', 'conv5_1.weight'), ('features.24.bias', 'conv5_1.bias'), ('features.26.weight', 'conv5_2.weight'), ('features.26.bias', 'conv5_2.bias'), ('features.28.weight', 'conv5_3.weight'), ('features.28.bias', 'conv5_3.bias')])

class VGG(nn.Module):

    def __init__(self, features, num_classes=200, init_weights=True, drop_thr=0.7):
        super(VGG, self).__init__()

        self.eil3 = EIL(drop_thr=0.7)
        self.eil4 = EIL(drop_thr=0.7)
        self.eil5 = EIL(drop_thr=0.7)

        self.cel3 = CEL(drop_thr=0.7)
        self.cel4 = CEL(drop_thr=0.7)
        self.cel5 = CEL(drop_thr=0.7)

        self.thr_val = drop_thr
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.center = nn.Parameter(torch.randn((num_classes, 1024)), requires_grad=False)
        self.counter = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

        # 64 x 224 x 224
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)

        # 64 x 224 x 224
        self.conv1_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)

        # 64 x 112 x 112
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 128 x 112 x 112
        self.conv2_1 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)

        # 128 x 112 x 112
        self.conv2_2 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

        # 128 x 56 x 56
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 256 x 56 x 56
        self.conv3_1 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)

        self.conv3_3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)

        # 256 x 28 x 28
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512 x 28 x 28
        self.conv4_1 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)

        self.conv4_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)

        self.conv4_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)

        #  512 x 14 x 14
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 512 x 14 x 14
        self.conv5_1 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)

        self.conv5_2 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)

        self.conv5_3 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)

        # above is backbone network
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

        # Weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x, **kwargs):
        ddt_obj_map_total = list()
        self.mode = kwargs["mode"]
        self.target = kwargs["target"]
        # self.target = kwargs["target"]
        x = self.conv1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)

        # below code --- don't touch any more

        x = self.conv2_1(x)
        x = self.relu2_1(x)

        x = self.conv2_2(x)
        x = self.relu2_2(x)

        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)

        x = self.conv3_2(x)
        x = self.relu3_2(x)

        x = self.conv3_3(x)
        x = self.relu3_3(x)

        x = self.pool3(x)

        self.pool3_fea = x
        x = self.conv4_1(x)
        x = self.relu4_1(x)

        x = self.conv4_2(x)
        x = self.relu4_2(x)

        x = self.conv4_3(x)
        x = self.relu4_3(x)

        x = self.pool4(x)

        # self.x = x
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        unerased = x
        self.unerased_51 = unerased

        unerased = self.conv5_2(unerased)
        unerased = self.relu5_2(unerased)
        self.unerased_52 = unerased

        unerased = self.conv5_3(unerased)
        unerased = self.relu5_3(unerased)
        self.unerased_53 = unerased

        # if self.mode == 'base':
        unerased = self.conv6(unerased)
        unerased = self.relu(unerased)

        self.feature_map = unerased  # 32，1024，14，14
        unerased = self.avgpool(unerased)  # 32，1024，1，1

        unerased = unerased.view(unerased.size(0), -1)  # 32，1024
        unerased = self.fc(unerased)  # 32，200
        pro_unerased, label = torch.max(unerased, dim=1)
        self.pro_unerased = pro_unerased
        self.label = label
        self.score = unerased

        if self.mode == 'train':
            cam, _ = self.get_fused_cam()
            self.center_list = self.get_class_center(cam)
            pool4 = self.eil5(x)
            # center_list.append(center)

        elif self.mode == 'train_with_erasing':
            #擦除主成分方向的区域
            cam, _ = self.get_fused_cam()
            self.center_list = self.get_class_center(cam)
            self.mask = kwargs["mask"]
            pool4 = self.cel5(self.unerased_52, self.mask)

        elif self.mode == 'eva_pca':
            #找到组成分方向，获得pca
            unerased_52 = self.unerased_51
            # unerased_52 = (1+4*unerased)*unerased_52
            for i in range(2):
                ddt_map = CEL.get_pca_by_first_compo(CEL,unerased_52, kwargs["compo"], self.label, self.mode)
                ddt_map_mask = self.erase_posi(ddt_map, self.thr_val)
                unerased_52 = self.erase_attention(unerased_52, ddt_map_mask)
                ddt_obj_map_total.append(ddt_map)
            self.ddt_obj_map_list = ddt_obj_map_total
            pool4 = self.eil4(x)

        elif self.mode == 'extra_compo':
            unerased = self.avgpool(self.unerased_53)  # 20，512，1，1
            unerased = normalize_tensor(unerased)
            # self.unerased_53 = (1+4*unerased)*self.unerased_53
            first_compo = get_ddt_obj(self.unerased_51, self.mode)
            self.first_compo = first_compo
            pool4 = self.eil4(x)

        elif self.mode == 'extra_pca':
            # 保存要擦除的位置信息
            unerased_53 = self.unerased_53
            for i in range(1):
                ddt_map = CEL.get_pca_by_first_compo(CEL, unerased_53, kwargs["compo"], self.label, self.mode)
                ddt_map_posi_a = self.erase_posi(ddt_map, self.thr_val) #要擦除的位置信息
                unerased_53 = self.erase_attention(unerased_53, ddt_map_posi_a)
                ddt_obj_map_total.append(ddt_map)

            self.ddt_obj_map_list = ddt_obj_map_total
            ddt_map = self.get_ddt_obj_list()
            ddt_map_posi = self.pca_posi(ddt_map, self.thr_val) #大于阈值则为1
            self.ddt_map_posi = ddt_map_posi
            pool4 = self.eil4(x)

        elif self.mode == 'eva':
            cam,_ = self.get_fused_cam()
            ddt_obj_map_total.append(cam)
            self.ddt_obj_map_list = ddt_obj_map_total
            pool4 = self.eil4(x)

        # pool4 = self.conv5_1(pool4)
        # pool4 = self.conv5_1(pool4)

        pool4 = self.conv5_2(pool4)
        pool4 = self.conv5_2(pool4)

        pool4 = self.conv5_3(pool4)
        pool4 = self.relu5_3(pool4)

        pool4 = self.conv6(pool4)
        pool4 = self.relu(pool4)

        pool4 = self.avgpool(pool4)
        pool4 = pool4.view(pool4.size(0), -1)

        pool4 = self.fc(pool4)#32 200
        pro_erased, _ = torch.max(pool4, dim=1)
        self.pro_erased = pro_erased
        self.score_erase4 = pool4

        return self.score

    def get_fused_cam(self):
        target = self.label
        if target is None:
            target = self.score.topk(1, 1, True, True)
        # if self.mode == 'base':
        batch, channel, _, _ = self.feature_map.size()
        fc_weight = self.fc.weight.squeeze()
        # print(self.fc.weight.shape,fc_weight.shape) torch.Size([200, 1024]) torch.Size([200, 1024])
        # get prediction in shape (batch)
        if target is None:
            _, target = self.score.topk(1, 1, True, True)

        target = target.squeeze()

        # get fc weight (num_classes x channel) -> (batch x channel)
        cam_weight = fc_weight[target]  #32 1024
        # get final cam with weighted sum of feature map and weights
        # (batch x channel x h x w) * ( batch x channel)
        cam_weight = cam_weight.view(
            batch, channel, 1, 1).expand_as(self.feature_map)
        cam = (cam_weight * self.feature_map)   #all are [32,1024, 14,14]
        cam = cam.mean(1)
        return cam.unsqueeze(1), self.score

    def get_ddt_map_posi(self):
        return self.ddt_map_posi

    def get_ddt_compo(self):
        return self.first_compo

    def get_feature(self, x):
        # input = Variable(torch.randn(1, 3, 224, 224))

        for index, layer in enumerate(self.features):
            x = layer(x)
            print(index, x.shape)
            # if index == selected_layer:
            #     return x
            # print(x)

    def get_ddt_obj_list(self):
        """
        return: return the obj ddt list base on mul [batch_size/gpu,1,h,w]
        """
        # merge torch.max()
        # self.ddt_obj_map_list.append(self.get_fused_cam()[0].cpu())
        
        if self.mode == 'train_with_erasing':
            cam,_ = self.get_fused_cam()
            ddt_map_posi_a = self.erase_posi(cam, self.thr_val)
            return ddt_map_posi_a
        # self.ddt_obj_map_list = normalize_tensor(self.ddt_obj_map_list)
        map_pro = (self.pro_unerased/self.pro_erased)*0.5
        self.ddt_obj_map_list[1] = self.ddt_obj_map_list[1].squeeze(1).cuda(0)
        map_pro = map_pro.unsqueeze(1).unsqueeze(1)
        # print(map_pro.device, self.ddt_obj_map_list[1].device)  #torch.Size([32]) torch.Size([32, 16, 16])

        # self.ddt_obj_map_list[1] = torch.mul(self.ddt_obj_map_list[1], map_pro)
        self.ddt_obj_map_list[1] = (self.ddt_obj_map_list[1] * map_pro).unsqueeze(1).cpu()
        ddt_total = torch.zeros_like(self.ddt_obj_map_list[0])

        for i in range(len(self.ddt_obj_map_list)):
            ddt_total = torch.max(ddt_total, self.ddt_obj_map_list[i])
        return ddt_total
        # self.ddt_obj_map_list.append(ddt_total)
        # return self.ddt_obj_map_list[0]

    def get_loss(self,  **kwargs):
        target = kwargs["target"]
        pca_maps = kwargs["mask"]
        batch_center_vecs = self.center_list
        self.center[target] = self.center[target].cuda()
        loss_center = torch.zeros_like(F.pairwise_distance(self.center[target].cuda().detach(),
                                          batch_center_vecs[0].cuda(), 2))
        self.update_center_vec(target, batch_center_vecs[0].cuda().detach())

        #设置擦除分类损失的权重参数
        loss_cls = self.CrossEntropyLoss(self.score_erase4, target) + self.CrossEntropyLoss(self.score, target)

        if self.mode == 'train':
            # return loss_cls
            return loss_cls + 0.6 * loss_center.mean(), loss_cls, loss_center.mean()
            # return loss, loss_cls, 0.02 * loss_center.mean()
        elif self.mode == 'eva':
            return self.CrossEntropyLoss(self.score_erase4, target) + self.CrossEntropyLoss(self.score, target)

        elif self.mode == 'train_with_erasing':

            pca = pca_maps.cuda().detach()
            pca.requires_grad = False
            cams = self.get_ddt_obj_list()
            batch, c, h, w = cams.size()
            loss_cams = (F.binary_cross_entropy(cams, pca.detach().float())) / batch
            del pca, pca_maps
            return loss_cls + 0.6 * loss_center.mean() + 0.5 * loss_cams.mean(),\
                   loss_cls,0.6 * loss_center.mean(),0.5 * loss_cams.mean()


    def update_center_vec(self, gt_labels, center_feats):
        lr = torch.exp(-0.002*self.counter[gt_labels])
        fused_centers = (1 - lr.detach()).unsqueeze(dim=1)*self.center[gt_labels].detach() \
                    + lr.detach().unsqueeze(dim=1)*center_feats.detach()
        self.counter[gt_labels] += 1
        self.center[gt_labels] = fused_centers

        del fused_centers

    def get_class_loss(self, target):
        return self.CrossEntropyLoss(self.score_erase4, target) + self.CrossEntropyLoss(self.score, target)

    def erase_attention(self, feature, mask):

        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        erased_feature = torch.mul(feature.type(Tensor), mask.type(Tensor))
        # add_attention = feature.type(Tensor) * attention_mask.type(Tensor)
        return erased_feature

    def erase_posi(self, attention_map, thr_val):
        b, _, h, w = attention_map.size()
        attention_map = attention_map.detach()
        pos = torch.ge(attention_map, thr_val)
        mask = attention_map.new_ones((b, 1, h, w))
        mask[pos.data] = 0.
        return mask

    def pca_posi(self, attention_map, thr_val):
        b, _, h, w = attention_map.size()
        attention_map = attention_map.detach()
        pos = torch.ge(attention_map, thr_val)
        mask = attention_map.new_zeros((b, 1, h, w))
        mask[pos.data] = 1.
        return mask

    def get_class_center(self, ddt_obj_map_pair_tensor):
        b, c, h, w = self.feature_map.size()
        center_list = torch.empty(b, c, h, w)
        for i in range(b):
            pos = torch.ge(ddt_obj_map_pair_tensor[i], torch.mean(ddt_obj_map_pair_tensor[i]))
            mask = torch.zeros((1, h, w)).fill_(0)
            mask[pos.data] = 1.
            cuda = True if torch.cuda.is_available() else False
            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            center = torch.mul(self.feature_map[i].type(Tensor), mask.type(Tensor))
            center_list[i] = center

        cls_center_feat = center_list.mean(dim=3).view(-1,1,c,h).mean(dim=3).mean(dim=1) #200 1024

        del pos, mask, center, center_list
        return cls_center_feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def normalize_tensor(x):
    map_size = x.size()  # 30,1,7,7  20,512,1,1
    aggregated = x.view(map_size[0], map_size[1], -1)  #  20,512,1,1
    minimum, _ = torch.min(aggregated, dim=1, keepdim=True)
    maximum, _ = torch.max(aggregated, dim=1, keepdim=True)
    normalized = torch.div(aggregated - minimum, maximum - minimum)  # div除法
    normalized = normalized.view(map_size)
    return normalized


def make_classifier(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(1024, out_planes, kernel_size=1, stride=1, padding=0),

        # nn.Conv2d(in_planes, 4096, kernel_size=3, stride=1, padding=1),
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.5, inplace=False),
        #
        # nn.Conv2d(4096, 4096, kernel_size=3, stride=1, padding=1),
        # nn.ReLU(inplace=True),
        # nn.Dropout(p=0.5, inplace=False),
        # nn.Conv2d(4096, out_planes, kernel_size=1, stride=1, padding=0),
    )


def make_layers(cfg, batch_norm=False):
    # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # MaxPool2d layers in ACoL
        elif v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'ACoL': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512, 512, 'M2', 512, 512, 512, 'M2'],
    'ACoL_1': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512, 512, 'M2', 512, 512, 'M2'],
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

def _vgg_ori(pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs['D_1']), **kwargs)
    if pretrained:
        state_dict = load_url(model_urls['vgg16'])
        state_dict = remove_layer(state_dict, 'classifier.')
        model.load_state_dict(state_dict, strict=False)
    return model

def _vgg_val(pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = models.vgg16(pretrained=True)
    if torch.cuda.is_available():
        model.cuda()
    if pretrained:
        state_dict = load_url(model_urls['vgg16'])
        model.load_state_dict(state_dict, strict=True)
    return model


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict

# def vgg16_acol(pretrained=True, **kwargs):
#
#     # return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)
#     return _vgg(pretrained, **kwargs)

def _vgg(pretrained=False, progress=True, **kwargs):
    kwargs['init_weights'] = True
    model = VGG(features=None, **kwargs)
    print(model)
    model_dict = model.state_dict()
    if pretrained:
        print("coming")
        pretrained_dict = models.vgg16(pretrained=True).state_dict()
        # state_dict = load_url(model_urls[arch], progress=progress)
        pretrained_dict = remove_layer(pretrained_dict, 'classifier.')

        for pretrained_k in pretrained_dict:
            if pretrained_k not in layer_mapping_vgg.keys():
                continue

            my_k = layer_mapping_vgg[pretrained_k]
            if my_k not in model_dict.keys():
                my_k = "module."+my_k
            if my_k not in model_dict.keys():
                raise Exception("Try to load not exist layer...")
            model_dict[my_k] = pretrained_dict[pretrained_k]
            # print("corresponding\t",my_k,"\t",pretrained_k)

    # model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(model_dict)
    return model