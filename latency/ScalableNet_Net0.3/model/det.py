# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from utils.image_utils import filter_border, nms, topk_map, get_gauss_filter_weight
from utils.image_utils import soft_nms_3d, soft_max_and_argmax_1d
from utils.math_utils import L2Norm


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class RFDet(nn.Module):
    def __init__(
        self,
        score_com_strength,
        scale_com_strength,
        nms_thresh,
        nms_ksize,
        topk,
        gauss_ksize,
        gauss_sigma,
        ksize,
        padding,
        dilation,
        scale_list,
    ):
        super(RFDet, self).__init__()

        self.score_com_strength = score_com_strength
        self.scale_com_strength = scale_com_strength
        self.NMS_THRESH = nms_thresh
        self.NMS_KSIZE = nms_ksize
        self.TOPK = topk
        self.GAUSSIAN_KSIZE = gauss_ksize
        self.GAUSSIAN_SIGMA = gauss_sigma

        self.features = []
        self.features.append(InvertedResidual(3, 12, 1, 2))
        self.features.append(InvertedResidual(12, 12, 1, 1))
        self.features = nn.Sequential(*self.features)
        ###########################################################################################################
        self.conv_1_1 = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=ksize, stride=1, padding=padding,
                                  dilation=dilation, )
        self.insnorm_1_1 = nn.InstanceNorm2d(1, affine=True)
        ############################################################################################################
        self.conv_scale = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.insnorm_scale = nn.InstanceNorm2d(1, affine=True)
        ############################################################################################################
        self.conv_ori = nn.Conv2d(in_channels=12, out_channels=2, kernel_size=1, stride=1, padding=0)

        self.scale_list = torch.tensor([12.0, 32.0, 64.0])

    def forward(self, photos):
        H, W = photos.shape[2:4]
        feature_map = self.features(photos)
        feature_map_2 = torch.nn.Upsample(scale_factor=0.75, mode='bilinear')(feature_map)
        feature_map_3 = torch.nn.Upsample(scale_factor=0.5, mode='bilinear')(feature_map)

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
        score_maps = score_maps.permute(0, 2, 3, 1)
        scale_probs = soft_nms_3d(score_maps, ksize=15, com_strength=10.0)
        score_map, scale_map = soft_max_and_argmax_1d(
            input=scale_probs,
            orint_maps=None,
            dim=-1,
            scale_list=self.scale_list,
            keepdim=True,
            com_strength1=self.score_com_strength,
            com_strength2=self.scale_com_strength,
        )
        ori_map = L2Norm(self.conv_ori(feature_map), dim=1).permute(0, 2, 3, 1)
        #import pdb
        #pdb.set_trace()
        return score_map, scale_map, ori_map

    def process(self, im1w_score):
        """
        nms(n), topk(t), gaussian kernel(g) operation
        :param im1w_score: warped score map
        :return: processed score map, topk mask, topk value
        """
        im1w_score = filter_border(im1w_score)

        # apply nms to im1w_score
        nms_mask = nms(im1w_score, thresh=self.NMS_THRESH, ksize=self.NMS_KSIZE)
        im1w_score = im1w_score * nms_mask
        topk_value = im1w_score

        # apply topk to im1w_score
        topk_mask = topk_map(im1w_score, self.TOPK)
        im1w_score = topk_mask.to(torch.float) * im1w_score

        # apply gaussian kernel to im1w_score
        psf = im1w_score.new_tensor(
            get_gauss_filter_weight(self.GAUSSIAN_KSIZE, self.GAUSSIAN_SIGMA)[
                None, None, :, :
            ]
        )
        im1w_score = F.conv2d(
            input=im1w_score.permute(0, 3, 1, 2),
            weight=psf,
            stride=1,
            padding=self.GAUSSIAN_KSIZE // 2,
        ).permute(
            0, 2, 3, 1
        )  # (B, H, W, 1)

        """
        apply tf.clamp to make sure all value in im1w_score isn't greater than 1
        but this won't happend in correct way
        """
        im1w_score = im1w_score.clamp(min=0.0, max=1.0)

        return im1w_score, topk_mask, topk_value

    @staticmethod
    def loss(left_score, im1gt_score, im1visible_mask):
        im1_score = left_score

        l2_element_diff = (im1_score - im1gt_score) ** 2
        # visualization numbers
        Nvi = torch.clamp(im1visible_mask.sum(dim=(3, 2, 1)), min=2.0)
        loss = (
            torch.sum(l2_element_diff * im1visible_mask, dim=(3, 2, 1)) / (Nvi + 1e-8)
        ).mean()

        return loss

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(
                m.weight.data, gain=nn.init.calculate_gain("leaky_relu")
            )
            try:
                nn.init.xavier_uniform_(m.bias.data)
            except:
                pass

    @staticmethod
    def convO_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight.data)
            try:
                nn.init.ones_(m.bias.data)
            except:
                pass
