import torch.nn as nn
import torch.nn.functional as F

class Center(nn.Module):
    def __init__(self):
        super(Center, self).__init__()



    def forward(self, center_list, batch_center_vecs, gt_labels):

        loss_center = F.pairwise_distance(center_list[gt_labels].cuda().detach(),
                                                        batch_center_vecs.cuda(), 2)
        self.update_center_vec(gt_labels, batch_center_vecs[i].cuda().detach())
        return loss_center.mean()

