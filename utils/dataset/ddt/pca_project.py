import torch.nn as nn
import torch
import numpy as np

class PCAProjectNet(nn.Module):
    def __init__(self):
        super(PCAProjectNet, self).__init__()


    def forward(self, features, mode):     # features: NCWH 16.200,14,14
        # # features  = features.unsqueeze(0)
        # # x_mean --> [200]
        # x_mean = (features.sum(dim=2).sum(dim=2).sum(dim=0) / k).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        # #torch.Size([1, 512, 1, 1])
        # features = features - x_mean
        # #reshaped_features -->torch.Size([512, 5880])
        # #features.view(features.size(0), features.size(1), -1).shape --> [30,512,196]
        # reshaped_features = features.view(features.size(0), features.size(1), -1)\
        #     .permute(1, 0, 2).contiguous().view(features.size(1), -1)
        # #cov.shape --> [512,512]
        # cov = torch.matmul(reshaped_features, reshaped_features.t()) / k
        # eigval, eigvec = torch.eig(cov, eigenvectors=True) #计算cov的特征值和特征向量
        # first_compo = eigvec[:, 0]  #取所有行的第0个数据，也就是取最大的方向 shape --> [512]
        # #torch.matmul(first_compo.unsqueeze(0), reshaped_features) --> [1,5880]
        # #.view(1, features.size(0), -1) --> [1,30,196]
        # #view(features.size(0), features.size(2), features.size(3)) [30,14,14]
        #
        # # projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1)\
        # #     .view(features.size(0), features.size(2), features.size(3))
        # # del eigval, eigvec,cov,reshaped_features,features,x_mean
        # #
        # # maxv = projected_map.max()
        # # minv = projected_map.min()
        # #
        # # projected_map *= (maxv + minv) / torch.abs(maxv + minv)
        k = features.size(0) * features.size(2) * features.size(3)  # N*W*H 1*14*14
        x_mean = (features.sum(dim=2).sum(dim=2).sum(dim=0) / k).unsqueeze(0).unsqueeze(2).unsqueeze(2)
        features = features - x_mean
        reshaped_features = features.view(features.size(0), features.size(1), -1) \
            .permute(1, 0, 2).contiguous().view(features.size(1), -1)
        cov = torch.matmul(reshaped_features, reshaped_features.t()) / k
        eigval, eigvec = torch.eig(cov, eigenvectors=True)  # 计算cov的特征值和特征向量
        first_compo = eigvec[:, 0]  # 取所有行的第0个数据，也就是取最大的方向 shape --> [512]
        if mode == 'extra_compo':
            return first_compo

        projected_map = torch.matmul(first_compo.unsqueeze(0), reshaped_features).view(1, features.size(0), -1) \
            .view(features.size(0), features.size(2), features.size(3))
        maxv = projected_map.max()
        minv = projected_map.min()
        projected_map *= (maxv + minv) / torch.abs(maxv + minv)
        return projected_map

if __name__ == '__main__':
    img = torch.randn(6, 512, 14, 14)
    pca = PCAProjectNet()
    pca(img)
