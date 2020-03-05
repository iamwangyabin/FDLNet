# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_utils import filter_border, nms, topk_map, get_gauss_filter_weight
from utils.image_utils import soft_nms_3d, soft_max_and_argmax_1d

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.features = []
        self.features.append(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1))
        self.features.append(InvertedResidual(16, 16, 1, 1))
        self.features.append(InvertedResidual(16, 16, 1, 1))
        self.features = nn.Sequential(*self.features)

    def forward(self, photos):
        feature_map = self.features(photos)

        feature_map_1 = F.leaky_relu(self.insnorm(self.conv(feature_map)))
        scale_probs = soft_nms_3d(score_maps, ksize=15, com_strength=7.0)
        score_map, scale_map = soft_max_and_argmax_1d(
            input=scale_probs,
            orint_maps=None,
            dim=-1,
            scale_list=self.scale_list,
            keepdim=True,
            com_strength1=self.score_com_strength,
            com_strength2=self.scale_com_strength,
        )
        return score_map, scale_map





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
        psf = get_gauss_filter_weight(self.GAUSSIAN_KSIZE, self.GAUSSIAN_SIGMA)[
              None, None, :, :
              ].to(im1w_score.device)

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

