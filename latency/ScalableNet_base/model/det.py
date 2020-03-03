# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image_utils import filter_border, nms, topk_map, get_gauss_filter_weight
from utils.image_utils import soft_nms_3d, soft_max_and_argmax_1d
from utils.math_utils import L2Norm

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

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 3 RF
        self.insnorm1 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s3 = nn.InstanceNorm2d(1, affine=True)

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 5 RF
        self.insnorm2 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s5 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s5 = nn.InstanceNorm2d(1, affine=True)

        self.conv3 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 7 RF
        self.insnorm3 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s7 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s7 = nn.InstanceNorm2d(1, affine=True)

        self.conv4 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 9 RF
        self.insnorm4 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s9 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s9 = nn.InstanceNorm2d(1, affine=True)

        self.conv5 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 11 RF
        self.insnorm5 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s11 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s11 = nn.InstanceNorm2d(1, affine=True)

        self.conv6 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 13 RF
        self.insnorm6 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s13 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s13 = nn.InstanceNorm2d(1, affine=True)

        self.conv7 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 15 RF
        self.insnorm7 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s15 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s15 = nn.InstanceNorm2d(1, affine=True)

        self.conv8 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 17 RF
        self.insnorm8 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s17 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s17 = nn.InstanceNorm2d(1, affine=True)

        self.conv9 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 19 RF
        self.insnorm9 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s19 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0 )
        self.insnorm_s19 = nn.InstanceNorm2d(1, affine=True)

        self.conv10 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=ksize,stride=1,padding=padding,dilation=dilation)  # 21 RF
        self.insnorm10 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s21 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s21 = nn.InstanceNorm2d(1, affine=True)

        self.conv_o3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o5 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o7 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o9 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o11 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o13 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o15 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o17 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o19 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o21 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)

        self.scale_list = torch.tensor(scale_list)


    def forward(self, photos):
        # Extract score map in scale space from 3 to 21
        score_featmaps_s3 = F.leaky_relu(self.insnorm1(self.conv1(photos)))
        score_map_s3 = self.insnorm_s3(self.conv_s3(score_featmaps_s3)).permute(
            0, 2, 3, 1
        )
        orint_map_s3 = (
            L2Norm(self.conv_o3(score_featmaps_s3), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )

        score_featmaps_s5 = F.leaky_relu(self.insnorm2(self.conv2(score_featmaps_s3)))
        score_map_s5 = self.insnorm_s5(self.conv_s5(score_featmaps_s5)).permute(
            0, 2, 3, 1
        )
        orint_map_s5 = (
            L2Norm(self.conv_o5(score_featmaps_s5), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s5 = score_featmaps_s5 + score_featmaps_s3

        score_featmaps_s7 = F.leaky_relu(self.insnorm3(self.conv3(score_featmaps_s5)))
        score_map_s7 = self.insnorm_s7(self.conv_s7(score_featmaps_s7)).permute(
            0, 2, 3, 1
        )
        orint_map_s7 = (
            L2Norm(self.conv_o7(score_featmaps_s7), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s7 = score_featmaps_s7 + score_featmaps_s5

        score_featmaps_s9 = F.leaky_relu(self.insnorm4(self.conv4(score_featmaps_s7)))
        score_map_s9 = self.insnorm_s9(self.conv_s9(score_featmaps_s9)).permute(
            0, 2, 3, 1
        )
        orint_map_s9 = (
            L2Norm(self.conv_o9(score_featmaps_s9), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s9 = score_featmaps_s9 + score_featmaps_s7

        score_featmaps_s11 = F.leaky_relu(self.insnorm5(self.conv5(score_featmaps_s9)))
        score_map_s11 = self.insnorm_s11(self.conv_s11(score_featmaps_s11)).permute(
            0, 2, 3, 1
        )
        orint_map_s11 = (
            L2Norm(self.conv_o11(score_featmaps_s11), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s11 = score_featmaps_s11 + score_featmaps_s9

        score_featmaps_s13 = F.leaky_relu(self.insnorm6(self.conv6(score_featmaps_s11)))
        score_map_s13 = self.insnorm_s13(self.conv_s13(score_featmaps_s13)).permute(
            0, 2, 3, 1
        )
        orint_map_s13 = (
            L2Norm(self.conv_o13(score_featmaps_s13), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s13 = score_featmaps_s13 + score_featmaps_s11

        score_featmaps_s15 = F.leaky_relu(self.insnorm7(self.conv7(score_featmaps_s13)))
        score_map_s15 = self.insnorm_s15(self.conv_s15(score_featmaps_s15)).permute(
            0, 2, 3, 1
        )
        orint_map_s15 = (
            L2Norm(self.conv_o15(score_featmaps_s15), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s15 = score_featmaps_s15 + score_featmaps_s13

        score_featmaps_s17 = F.leaky_relu(self.insnorm8(self.conv8(score_featmaps_s15)))
        score_map_s17 = self.insnorm_s17(self.conv_s17(score_featmaps_s17)).permute(
            0, 2, 3, 1
        )
        orint_map_s17 = (
            L2Norm(self.conv_o17(score_featmaps_s17), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s17 = score_featmaps_s17 + score_featmaps_s15

        score_featmaps_s19 = F.leaky_relu(self.insnorm9(self.conv9(score_featmaps_s17)))
        score_map_s19 = self.insnorm_s19(self.conv_s19(score_featmaps_s19)).permute(
            0, 2, 3, 1
        )
        orint_map_s19 = (
            L2Norm(self.conv_o19(score_featmaps_s19), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )
        score_featmaps_s19 = score_featmaps_s19 + score_featmaps_s17

        score_featmaps_s21 = F.leaky_relu(
            self.insnorm10(self.conv10(score_featmaps_s19))
        )
        score_map_s21 = self.insnorm_s21(self.conv_s21(score_featmaps_s21)).permute(
            0, 2, 3, 1
        )
        orint_map_s21 = (
            L2Norm(self.conv_o21(score_featmaps_s21), dim=1)
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
        )

        score_maps = torch.cat(
            (
                score_map_s3,
                score_map_s5,
                score_map_s7,
                score_map_s9,
                score_map_s11,
                score_map_s13,
                score_map_s15,
                score_map_s17,
                score_map_s19,
                score_map_s21,
            ),
            -1,
        )  # (B, H, W, C)

        orint_maps = torch.cat(
            (
                orint_map_s3,
                orint_map_s5,
                orint_map_s7,
                orint_map_s9,
                orint_map_s11,
                orint_map_s13,
                orint_map_s15,
                orint_map_s17,
                orint_map_s19,
                orint_map_s21,
            ),
            -2,
        )  # (B, H, W, 10, 2)

        # get each pixel probability in all scale
        scale_probs = soft_nms_3d(score_maps, ksize=15, com_strength=3.0)
        # import pdb
        # pdb.set_trace()
        # get each pixel probability summary from all scale space and correspond scale value
        score_map, scale_map, orint_map = soft_max_and_argmax_1d(
            input=scale_probs,
            orint_maps=orint_maps,
            dim=-1,
            scale_list=self.scale_list,
            keepdim=True,
            com_strength1=self.score_com_strength,
            com_strength2=self.scale_com_strength,
        )

        return score_map, scale_map, orint_map

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
