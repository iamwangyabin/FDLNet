# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from model.shufflenetmodule import InvertedResidual

class RFDet(nn.Module):
    def __init__(self):
        super(RFDet, self).__init__()
        self.features = []
        self.features.append(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        self.features.append(InvertedResidual(16, 16, 1, 1))
        self.features = nn.Sequential(*self.features)

        self.conv_1_1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1,)
        self.insnorm_1_1 = nn.InstanceNorm2d(1, affine=True)

        self.conv_ori = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.insnorm_ori = nn.InstanceNorm2d(2, affine=True)

        self.scale_list = torch.tensor([16.0, 32.0, 64.0])

    def forward(self, photos):
        H, W = photos.shape[2:4]
        feature_map = self.features(photos)
        feature_map_2 = torch.nn.Upsample(scale_factor=0.75, mode='bilinear')(feature_map)
        feature_map_3 = torch.nn.Upsample(scale_factor=0.5, mode='bilinear')(feature_map)
        ori_map = L2Norm(torch.nn.Upsample(size=(H, W), mode='bilinear')(self.insnorm_ori(self.conv_ori(feature_map_3))), dim=1).permute(0, 2, 3, 1)

        feature_map_1 = F.leaky_relu(self.insnorm_1_1(self.conv_1_1(feature_map)))
        feature_map_2 = F.leaky_relu(self.insnorm_1_1(self.conv_1_1(feature_map_2)))
        feature_map_3 = F.leaky_relu(self.insnorm_1_1(self.conv_1_1(feature_map_3)))

        score_maps = torch.cat(
            (
                feature_map_1,
                torch.nn.Upsample(size=(H, W), mode='bilinear')(feature_map_2),
                torch.nn.Upsample(size=(H, W), mode='bilinear')(feature_map_3),
            ),
            1,
        )  # (B, C, H, W)
        return score_maps,ori_map



input_sample = torch.autograd.Variable(torch.randn((1, 1, 600, 460)).to(torch.float32)).cuda()

model = RFDet().cuda()
import time
time0=time.time()
for i in range(100):
    l,o= model(input_sample.cuda())
time1=time.time()
print((time1-time0)/100)

# time0=time.time()
# for i in range(100):
#     torch.nn.Upsample(size=(300, 200), mode='bilinear')(input_sample)
# time1=time.time()
# print((time1-time0)/100)
#
#
#
# time0=time.time()
# for i in range(100):
#     score_maps = torch.cat(
#         (
#             input_sample,
#             input_sample,
#             input_sample,
#         ),
#         1,
#     )  # (B, C, H, W)
#
# time1=time.time()
# print((time1-time0)/100)

from model.netmodule import IRFBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.math_utils import L2Norm
from model.shufflenetmodule import InvertedResidual

class HardNetNeiMask(nn.Module):
    def __init__(self):
        super(HardNetNeiMask, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # InvertedResidual(32, 32, 1, 1),
            # InvertedResidual(32, 64, 2, 2),
            # InvertedResidual(64, 64, 1, 1),
            # InvertedResidual(64, 128, 2, 2),
            # IRFBlock(32, 32, 0.5, 2, kernel=3),
            IRFBlock(32, 64, 0.5, 2, kernel=3),
            IRFBlock(64, 128, 0.5, 1, kernel=3),
            IRFBlock(128, 128, 0.1, 2, kernel=3),
            nn.Conv2d(128, 128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    @staticmethod
    def input_norm(x):
        eps = 1e-8
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + eps
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        feature = x / torch.norm(x, p=2, dim=-1, keepdim=True)
        return feature

input_sample = torch.autograd.Variable(torch.randn((1000, 1, 32, 32)).to(torch.float32)).cuda()
model = HardNetNeiMask().cuda()
import time
time0=time.time()
for i in range(100):
    l= model(input_sample.cuda())
time1=time.time()
print((time1-time0)/100)