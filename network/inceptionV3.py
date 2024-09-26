import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.model_zoo import load_url
import torch.utils.model_zoo as model_zoo

import torchvision.models as models
import numpy as np
from utils.ddt.ddt_func import *
from torch.autograd import Variable
from utils.ddt.pca_project import *
from network.inceptionV3_sub import *

__all__ = [
    'InceptionV3', 'inceptionV3_cel',
]

model_urls = {
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class InceptionV3(nn.Module):

    def __init__(self, features, num_classes=200, init_weights=True, drop_thr=0.8):
        super(InceptionV3, self).__init__()
        self.features = features
        self.classifier_A = make_classifier(768, num_classes)
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)

        if init_weights:
            self._initialize_weights()
        self.thr_val = drop_thr
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.feature_extra = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),
            InceptionB(288, kernel_size=3, stride=1, padding=1),
            InceptionC(768, channels_7x7=128),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=192),
        )
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # # Added
        # self.cls_fc6 = nn.Sequential(
        #     nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),
        #     nn.ReLU(True),
        # )
        # self.cls_fc7 = nn.Sequential(
        #     nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
        #     nn.ReLU(True))
        # self.cls_fc8 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)
        self.center = nn.Parameter(torch.randn((num_classes, 768)), requires_grad=False)
        self.counter = nn.Parameter(torch.zeros(num_classes), requires_grad=False)


    def forward(self, x, compo_or_pca, target, mode):
        logits = list()
        ddt_obj_map_total = list()
        center_list = list()
        first_compo_list = list()
        # feature_list = list()
        self.mode = mode

        #299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)  #150 x 150 x 32
        x = self.Conv2d_2a_3x3(x)  #148 x 148 x 32
        x = self.Conv2d_2b_3x3(x)  #148 x 148 x 64
        feat1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True) #150 x 150 x 32
        x = self.Conv2d_3b_1x1(feat1)  #75 x 75 x 64
        x = self.Conv2d_4a_3x3(x)  #75 x 75 x 80
        feat2 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True) #73 x 73 x 192
        x = self.Mixed_5b(feat2)   #37 x 37 x 192
        x = self.Mixed_5c(x)       #37 x 37 x 256
        feat3 = self.Mixed_5d(x)   #37 x 37 x 288

        x = self.Mixed_6a(feat3)   #37 x 37 x 288
        x = self.Mixed_6b(x)      #37 x 37 x 768
        x = self.Mixed_6c(x)      #37 x 37 x 768
        x = self.Mixed_6d(x)      #37 x 37 x 768
        self.feature_ori = self.Mixed_6e(x)   #37 x 37 x 768

        # Branch 1
        # out1, last_feat = self.inference(feat4)
        # logits_1 = torch.mean(torch.mean(out1, dim=2), dim=2)
        # self.map1 = out1

        # return [logits_1, self.get_localization_maps()]
        # return [logits_1, ]

        # self.feature_ori = self.features(x)
        if mode == 'train':
            for i in range(1):
                self.feature_ddt = self.classifier_A(self.feature_ori)  # 200个通道，每个通道对应不同的特征图
                logit_a = self.GlobalAvgPool(self.feature_ddt)
                b, c, _, _ = logit_a.size()
                logit_a = logit_a.view(b, c)
                logits.append(logit_a)
                self.score = logit_a
                _, label = torch.max(logits[0], dim=1)
                self.label = label

                ddt_obj_map = self.get_cam()
                center = self.get_class_center(ddt_obj_map)
                center_list.append(center)
                ddt_obj_map_total.append(ddt_obj_map)

                # attention = self.get_attention()
                # self.feature_ori, mask = self.erase_attention(self.feature_ori, attention, self.thr_val)

                del ddt_obj_map
                return logits, center_list, ddt_obj_map_total

        if mode =='train_with_erasing':
            for i in range(1):
                self.feature_ddt = self.classifier_A(self.feature_ori)  # 200个通道，每个通道对应不同的特征图
                logit_a = self.GlobalAvgPool(self.feature_ddt)
                b, c, _, _ = logit_a.size()
                logit_a = logit_a.view(b, c)
                logits.append(logit_a)
                self.score = logit_a
                _, label = torch.max(logits[0], dim=1)
                self.label = label

                ddt_obj_map = compo_or_pca
                center = self.get_class_center(ddt_obj_map)
                ddt_obj_map_total.append(ddt_obj_map)
                center_list.append(center)

                # self.feature_ori, mask = self.erase_attention(self.feature_ori, ddt_obj_map, self.thr_val)

                del  ddt_obj_map
                return logits, center_list, ddt_obj_map_total

        if mode == 'eva':
            for i in range(3):
                self.feature_ddt = self.classifier_A(self.feature_ori)  # 200个通道，每个通道对应不同的特征图
                logit_a = self.GlobalAvgPool(self.feature_ddt)
                b, c, _, _ = logit_a.size()
                logit_a = logit_a.view(b, c)
                logits.append(logit_a)
                _, label = torch.max(logits[0], dim=1)
                self.label = label

                attention = self.get_attention()
                self.feature_ori, mask = self.erase_attention(self.feature_ori, attention, self.thr_val)
                ddt_obj_map_total.append(attention)
            self.ddt_obj_map_list = ddt_obj_map_total
            return logits

        self.feature_ddt = self.classifier_A(self.feature_ori)  # 200个通道，每个通道对应不同的特征图
        logit_a = self.GlobalAvgPool(self.feature_ddt)
        b, c, _, _ = logit_a.size()
        logit_a = logit_a.view(b, c)
        logits.append(logit_a)

        # #提取主成分
        if mode == 'extra_compo':
            _, label = torch.max(logit_a, dim=1)
            self.label = label
            first_compo = self.get_ddt_obj(self.feature_ddt, mode)
            return logits, first_compo

        #保存要擦除的位置信息
        if mode == 'extra_pca':
            self.score = logit_a
            ddt_map = self.get_pca_by_first_compo(compo_or_pca, target)
            return logits, ddt_map

    def inference(self, x):
        if self.training:
            x = F.dropout(x, 0.5)
        x = self.cls_fc6(x)

        if self.training:
            x = F.dropout(x, 0.5)
        x = self.cls_fc7(x)

        if self.training:
            x = F.dropout(x, 0.5)
        out1 = self.cls_fc8(x)

        return out1, x

    def get_cam(self):
        """
        :return: return attention size (batch, 1, h, w)
        返回每个类中阈值大于0.5的特征图
        """
        cams = normalize_tensor(self.feature_ddt)
        batch, channel, _, _ = cams.size()

        _, target = self.score.topk(1, 1, True, True)
        target = target.squeeze()

        cam = cams[range(batch), target] # target 表明每个类中得分最高的那个cam图
        del cams

        return cam.unsqueeze(1)

    def get_attention(self, normalize=True):
        """
        :return: return attention size (batch, 1, h, w)
        """
        label = self.label.long()
        b = self.feature_ddt.size(0)
        attention = self.feature_ddt.detach().clone().requires_grad_(True)[range(b), label.data, :, :]
        attention = attention.unsqueeze(1)
        if normalize:
            attention = normalize_tensor(attention)
        return attention

    def get_ddt_obj(self, feat_map_a, mode):
        """
        return: return the obj ddt find [batch_size/gpu,1,h,w]
        """
        pca = PCAProjectNet()
        project_map_or_first_compo = pca(feat_map_a, mode)
        if mode == 'extra_compo':
            return project_map_or_first_compo
        b,_,h,w = self.feature_ddt.size()
        project_map = torch.clamp(project_map_or_first_compo, min=0.01)
        maxv = project_map.view(project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_map /= maxv
        project_map = F.interpolate(project_map.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        pro = project_map.squeeze(1)
        one = torch.ones(h, w).type(Tensor)
        for i in range(project_map.size(0)):
            pos = torch.equal(pro[0:, ][i], one)
            if pos:
                pro[0:, ][i] = 0
        project_map = pro.unsqueeze(1)
        project_map = project_map.detach()
        del pca, pro, one
        return project_map

    def get_ddt_obj_list(self):
        """
        return: return the obj ddt list base on mul [batch_size/gpu,1,h,w]
        """
        # merge torch.max()
        ddt_total = torch.zeros_like(self.ddt_obj_map_list[0])
        for i in range(len(self.ddt_obj_map_list)):
            ddt_total = torch.max(ddt_total, self.ddt_obj_map_list[i])
        return ddt_total

    def get_loss(self, output, gt_labels, pca_maps):

        cls_logits, batch_center_vecs, cams = output
        self.center[gt_labels] = self.center[gt_labels].cuda()
        loss_cams = torch.zeros_like(cams[0][0])
        loss_center = torch.zeros([gt_labels.size()[0]])
        loss_cls = torch.zeros_like(self.CrossEntropyLoss(cls_logits[0], gt_labels.long()))

        for i in range(len(cls_logits)):

            loss_center = loss_center.cuda() + F.pairwise_distance(self.center[gt_labels].cuda().detach(),
                                                                   batch_center_vecs[i].cuda(), 2)
            self.update_center_vec(gt_labels, batch_center_vecs[i].cuda().detach())
            loss_cls = loss_cls + (self.CrossEntropyLoss(cls_logits[i], gt_labels.long()))

            if self.mode == 'train':
                loss = loss_cls + 0.02 * loss_center.mean()
                return loss, loss_cls, 0.02 * loss_center.mean()

            if self.mode=='train_with_erasing':
                pca = pca_maps.cuda().detach()
                pca.requires_grad = False
                batch, c, h, w = cams[0].size()
                loss_cams = loss_cams.cuda() + (F.binary_cross_entropy(cams[0], pca.detach()) +
                                                F.binary_cross_entropy(1.0 - cams[0], 1.0 - pca.detach())) / (h * w) / batch

                del pca, pca_maps
                loss = loss_cls + 0.02 * loss_center.mean() + 20*loss_cams.mean()
                return loss, loss_cls, 0.02*loss_center.mean(), 20*loss_cams.mean()

    def update_center_vec(self, gt_labels, center_feats):
        lr = torch.exp(-0.002*self.counter[gt_labels])
        fused_centers = (1 - lr.detach()).unsqueeze(dim=1)*self.center[gt_labels].detach() \
                    + lr.detach().unsqueeze(dim=1)*center_feats.detach()
        self.counter[gt_labels] += 1
        self.center[gt_labels] = fused_centers

        del fused_centers

    def get_class_loss(self, logits, gt_labels):
        return self.CrossEntropyLoss(logits[0], gt_labels.long())

    def erase_attention(self, feature, attention_map, thr_val):
        b, _, h, w = attention_map.size()
        attention_map = attention_map.detach()
        pos = torch.ge(attention_map, thr_val)
        mask = attention_map.new_ones((b, 1, h, w))
        mask[pos.data] = 0.
        attention_mask = attention_map.new_zeros((b, 1, h, w))
        attention_mask[pos.data] = 1.
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        erased_feature = torch.mul(feature.type(Tensor), mask.type(Tensor))
        add_attention = feature.type(Tensor) * attention_mask.type(Tensor)

        del add_attention, attention_map, pos, attention_mask

        return erased_feature, mask

    # def get_ddt_map_pair(self):
    #     b, _, h, w = self.feature_ddt.size()
    #     ddt_obj_map_pair_tensor = torch.empty(1,1,h,w)
    #     feature_list = torch.split(self.feature_ddt, 2, dim=0)
    #     for feature in feature_list:
    #         ddt_obj_map_pair = self.get_ddt_obj(feature)
    #         ddt_obj_map_pair = ddt_obj_map_pair.to(ddt_obj_map_pair_tensor.device)
    #         ddt_obj_map_pair_tensor = torch.cat((ddt_obj_map_pair_tensor,ddt_obj_map_pair), dim=0)
    #     del feature_list, ddt_obj_map_pair
    #     return ddt_obj_map_pair_tensor[1:]

    def get_class_center(self, ddt_obj_map_pair_tensor):
        b, c, h, w = self.feature_ori.size()
        center_list = torch.empty(b, c, h, w)
        for i in range(b):
            pos = torch.ge(ddt_obj_map_pair_tensor[i], torch.mean(ddt_obj_map_pair_tensor[i]))
            mask = torch.zeros((1, h, w)).fill_(0).cuda()
            mask[pos.data] = 1.
            cuda = True if torch.cuda.is_available() else False
            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            center = torch.mul(self.feature_ori[i].type(Tensor), mask.type(Tensor))
            center_list[i] = center

        cls_center_feat = center_list.mean(dim=3).view(-1,1,c,h).mean(dim=3).mean(dim=1)

        del pos, mask, center, center_list
        return cls_center_feat

    def get_pca_by_first_compo(self, first_compo_list, target):

        b, c, h, w = self.feature_ddt.size()
        k = self.feature_ddt.size(2) * self.feature_ddt.size(3)  # N*W*H 1*14*14
        project_map_list = torch.zeros(b, 1, h, w)
        target = target.cpu().numpy()
        for i in range(b):
            features = self.feature_ddt[i].unsqueeze(0)
            x_mean = (features.sum(dim=2).sum(dim=2).sum(dim=0) / k).unsqueeze(0).unsqueeze(2).unsqueeze(2)
            features = features - x_mean
            reshaped_features = features.view(features.size(0), features.size(1), -1) \
                .permute(1, 0, 2).contiguous().view(features.size(1), -1)
            first_compo_idx = str(target[i])
            projected_map = torch.matmul(first_compo_list[first_compo_idx].unsqueeze(0),
                                         reshaped_features).view(1, features.size(0), -1) \
                .view(features.size(0), features.size(2), features.size(3))
            maxv = projected_map.max()
            minv = projected_map.min()
            projected_map *= (maxv + minv) / torch.abs(maxv + minv)
            project_map = self.first_compo_pca(projected_map)
            project_map_list[i] = project_map
        return project_map_list

    def first_compo_pca(self, project_map):
        b, _, h, w = self.feature_ddt.size()
        project_map = torch.clamp(project_map, min=0.01)
        maxv = project_map.view(project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
        project_map /= maxv
        project_map = F.interpolate(project_map.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False)
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        pro = project_map.squeeze(1)
        one = torch.ones(h, w).type(Tensor)
        for i in range(project_map.size(0)):
            pos = torch.equal(pro[0:, ][i], one)
            if pos:
                pro[0:, ][i] = 0
        project_map = pro.unsqueeze(1)
        return project_map

    def get_ddt_map_total_2(self,first_compo_list, target):

        # 根据特征明显的区域得到ddt位置
        ddt_obj_map = self.get_pca_by_first_compo(self.feature_ddt,first_compo_list, target)
        # feature, mask = self.erase_attention(self.feature_ori, ddt_obj_map, self.thr_val)
        return ddt_obj_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def normalize_tensor(x):
    map_size = x.size()  # 30,1,7,7
    aggregated = x.view(map_size[0], map_size[1], -1)  # 30,1,49
    minimum, _ = torch.min(aggregated, dim=-1, keepdim=True)
    maximum, _ = torch.max(aggregated, dim=-1, keepdim=True)
    normalized = torch.div(aggregated - minimum, maximum - minimum)  # div除法
    normalized = normalized.view(map_size)
    return normalized


def make_classifier(in_planes, out_planes):
    return nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),
        nn.ReLU(True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
        nn.ReLU(True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(1024, out_planes, kernel_size=1, padding=0),
    )
def make_layer(new_model):
    return new_model

def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict

def _inceptionV3(pretrained, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = models.inception_v3(pretrained=True)
    model.aux_logits=False
    model = nn.Sequential(*list(model.children())[:-1])
    # print(model)

    new_model = InceptionV3(make_layer(model))

    if torch.cuda.is_available():
        new_model.cuda()
    return new_model
    # if pretrained:
    #     state_dict = load_url(model_urls['inception_v3_google'])
    #     state_dict = remove_layer(state_dict, 'fc.')
    #     new_model.load_state_dict(state_dict, strict=True)
    # if torch.cuda.is_available():
    #     new_model.cuda()
    # return new_model

def inceptionV3_cel(pretrained=True, **kwargs):
    r"""inceptionV3 model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _inceptionV3(pretrained, **kwargs)