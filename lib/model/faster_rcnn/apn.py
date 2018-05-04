import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class MotionVector(nn.Module):
    def __init__(self, stride=16):
        super(MotionVector, self).__init__()
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.stride = int(stride / 2)

    def forward(self, img_cur, img_ref):
        img_ref = img_ref.cuda()
        img_cur = img_cur.cuda()

        img_ref = self.avg_pool(img_ref)
        img_cur = self.avg_pool(img_cur)

        batch_size, channel, height, width = img_cur.shape
        mv_height = int(height / self.stride)
        mv_width = int(width / self.stride)

        output = torch.zeros(batch_size, 2, mv_height, mv_width) # 2 means x-axis, y-axis
        return output

class GridGenerator(nn.Module):
    def __init__(self):
        super(GridGenerator, self).__init__()

    def forward(self, motion_vector):
        flow = motion_vector
        horizontal = torch.linspace(-1.0, 1.0, flow.size(3)).view(1, 1, 1, flow.size(3)).expand(flow.size(0), 1, flow.size(2), flow.size(3))
        vertical = torch.linspace(-1.0, 1.0, flow.size(2)).view(1, 1, flow.size(2), 1).expand(flow.size(0), 1, flow.size(2), flow.size(3))

        grid = torch.cat([ horizontal, vertical ], 1)
        flow = torch.cat([ flow[:, 0:1, :, :] / ((flow.size(3) - 1.0) / 2.0), flow[:, 1:2, :, :] / ((flow.size(2) - 1.0) / 2.0) ], 1)

        output = grid+flow
        output = output.permute(0, 2, 3, 1)

        return output

class APN(nn.Module):
    def __init__(self, stride=16):
        super(APN, self).__init__()
        self.MotionVector = MotionVector(stride)
        self.GridGenerator = GridGenerator()

    def forward(self, img_cur, img_ref, key_feat):
        # img_cur (N, C, H, W)
        # img_ref (N, C, H, W)
        # key_feat (N, C, H, W)

        motion_vector = self.MotionVector(img_cur, img_ref) # return (N, 2, H, W)
        grid = self.GridGenerator(motion_vector) # return (N, H, W, 2)

        key_feat = key_feat.cuda()
        grid = grid.cuda()

        output = F.grid_sample(key_feat, grid.clamp(-1, 1)) # return bilinear interpolation with motion vector
        return output
