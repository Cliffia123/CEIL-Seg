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

class CEL(nn.Module):
    def __init__(self, drop_rate=0.7, drop_thr=0.9):
        super(CEL, self).__init__()
        assert 0 <= drop_rate <= 1 and 0 <= drop_thr <= 1
        self.drop_rate = drop_rate
        self.drop_thr = drop_thr

    def extra_repr(self):
        return 'drop_rate={}, drop_thr={}'.format(
            self.drop_rate, self.drop_thr
        )
    def forward(self, x, pca_maps):
        feature_ori = self.erase_attention(x, pca_maps)
        return feature_ori

    def erase_attention(self, feature, mask):
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        b, _, h, w = mask.size()
        mask = mask.detach()
        pos = torch.ge(mask, 0.7)
        mask = mask.new_ones((b, 1, h, w))
        mask[pos.data] = 0.

        erased_feature = torch.mul(feature.type(Tensor), mask.type(Tensor))
        return erased_feature

    def get_pca_by_first_compo(self, feature_compo, first_compo_list, target, mode):

        b, c, h, w = feature_compo.size()
        k = feature_compo.size(2) * feature_compo.size(3)  # N*W*H 1*14*14
        project_map_list = torch.zeros(b, 1, h, w)
        target = target.cpu().numpy()
        for i in range(b):
            features = feature_compo[i].unsqueeze(0)
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
            project_map = first_compo_pca(projected_map)
            project_map_list[i] = project_map
        return project_map_list

def first_compo_pca(project_map):
    c, h, w = project_map.size()
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


def get_ddt_obj(feat_map_a, mode):
    """
    return: return the obj ddt find [batch_size/gpu,1,h,w]
    """
    pca = PCAProjectNet()
    b, _, w, h = feat_map_a.size()
    project_map_or_first_compo = pca(feat_map_a, mode)
    if mode == 'extra_compo':
        return project_map_or_first_compo

    project_map = torch.clamp(project_map_or_first_compo, min=0.01)
    maxv = project_map.view(project_map.size(0), -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
    project_map /= maxv
    project_map = F.interpolate(project_map.unsqueeze(1), size=(w, h), mode='bilinear', align_corners=False)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    pro = project_map.squeeze(1)
    one = torch.ones(w, h).type(Tensor)
    for i in range(project_map.size(0)):
        pos = torch.equal(pro[0:, ][i], one)
        if pos:
            pro[0:, ][i] = 0
    project_map = pro.unsqueeze(1)
    project_map = project_map.detach()
    del pca, pro, one
    return project_map