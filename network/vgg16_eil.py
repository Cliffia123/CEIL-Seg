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

class EIL(nn.Module):
    def __init__(self, drop_rate=0.75, drop_thr=0.9):
        super(EIL, self).__init__()
        assert 0 <= drop_rate <= 1 and 0 <= drop_thr <= 1
        self.drop_rate = drop_rate
        self.drop_thr = drop_thr

        self.attention = None
        self.drop_mask = None

    def extra_repr(self):
        return 'drop_rate={}, drop_thr={}'.format(
            self.drop_rate, self.drop_thr
        )

    def forward(self, x):
        b = x.size(0)
        attention = torch.mean(x, dim=1, keepdim=True)
        self.attention = attention
        max_val, _ = torch.max(attention.view(b, -1), dim=1, keepdim=True)
        thr_val = max_val * self.drop_thr
        thr_val = thr_val.view(b, 1, 1, 1).expand_as(attention)
        drop_mask = (attention < thr_val).float()
        self.drop_mask = drop_mask
        output = x.mul(drop_mask)
        return output

    def get_maps(self):
        return self.attention, self.drop_mask

