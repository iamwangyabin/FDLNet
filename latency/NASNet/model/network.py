# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from utils.common_utils import imgBatchXYZ, transXYZ_2_to_1
from utils.image_utils import warp, filter_border
from utils.math_utils import MSD, distance_matrix_vector, L2Norm
from utils.net_utils import pair
from utils.image_utils import clip_patch, topk_map

class Network(nn.Module):
    def __init__(self, det, des, SCORE_W, PAIR_W, PSIZE, TOPK):
        super(Network, self).__init__()
        self.det = det
        self.des = des

        self.PSIZE = PSIZE
        self.TOPK = TOPK
        self.SCORE_W = SCORE_W
        self.PAIR_W = PAIR_W

    def forward(self, batch):
        im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = batch
        im1_rawsc, im1_scale, im1_orin = self.det(im1_data)
        im2_rawsc, im2_scale, im2_orin = self.det(im2_data)

        im1_gtscale, im1_gtorin = self.gt_scale_orin(
            im2_scale, im2_orin, homo12, homo21
        )
        im2_gtscale, im2_gtorin = self.gt_scale_orin(
            im1_scale, im1_orin, homo21, homo12
        )

        im1_gtsc, im1_topkmask, im1_topkvalue, im1_visiblemask = self.gtscore(
            im2_rawsc, homo12
        )
        im2_gtsc, im2_topkmask, im2_topkvalue, im2_visiblemask = self.gtscore(
            im1_rawsc, homo21
        )
        im1_score = self.det.process(im1_rawsc)[0]
        im2_score = self.det.process(im2_rawsc)[0]

        ###############################################################################
        # Extract patch and its descriptors by corresponding scale and orination
        ###############################################################################
        # (B*topk, 2, 32, 32)
        im1_ppair, im1_limc, im1_rimcw = pair(
            im1_topkmask,
            im1_topkvalue,
            im1_scale,
            im1_orin,
            im1_info,
            im1_raw,
            homo12,
            im2_gtscale,
            im2_gtorin,
            im2_info,
            im2_raw,
            self.PSIZE,
        )
        im2_ppair, im2_limc, im2_rimcw = pair(
            im2_topkmask,
            im2_topkvalue,
            im2_scale,
            im2_orin,
            im2_info,
            im2_raw,
            homo21,
            im1_gtscale,
            im1_gtorin,
            im1_info,
            im1_raw,
            self.PSIZE,
        )

        im1_lpatch, im1_rpatch = im1_ppair.chunk(chunks=2, dim=1)  # each is (N, 32, 32)
        im2_lpatch, im2_rpatch = im2_ppair.chunk(chunks=2, dim=1)  # each is (N, 32, 32)

        im1_lpdes, im1_rpdes = self.des(im1_lpatch), self.des(im1_rpatch)
        im2_lpdes, im2_rpdes = self.des(im2_lpatch), self.des(im2_rpatch)
        ###############################################################################
        # Extract patch and its descriptors by predicted scale and orination
        ###############################################################################
        # (B*topk, 2, 32, 32)
        im1_predpair, _, _ = pair(
            im1_topkmask,
            im1_topkvalue,
            im1_scale,
            im1_orin,
            im1_info,
            im1_raw,
            homo12,
            im2_scale,
            im2_orin,
            im2_info,
            im2_raw,
            self.PSIZE,
        )
        im2_predpair, _, _ = pair(
            im2_topkmask,
            im2_topkvalue,
            im2_scale,
            im2_orin,
            im2_info,
            im2_raw,
            homo21,
            im1_scale,
            im1_orin,
            im1_info,
            im1_raw,
            self.PSIZE,
        )

        # each is (N, 32, 32)
        im1_lpredpatch, im1_rpredpatch = im1_predpair.chunk(chunks=2, dim=1)
        im2_lpredpatch, im2_rpredpatch = im2_predpair.chunk(chunks=2, dim=1)

        im1_lpreddes, im1_rpreddes = self.des(im1_lpredpatch), self.des(im1_rpredpatch)
        im2_lpreddes, im2_rpreddes = self.des(im2_lpredpatch), self.des(im2_rpredpatch)
        endpoint = {
            "im1_score": im1_score,
            "im1_gtsc": im1_gtsc,
            "im1_visible": im1_visiblemask,
            "im2_score": im2_score,
            "im2_gtsc": im2_gtsc,
            "im2_visible": im2_visiblemask,
            "im1_limc": im1_limc,
            "im1_rimcw": im1_rimcw,
            "im2_limc": im2_limc,
            "im2_rimcw": im2_rimcw,
            "im1_lpdes": im1_lpdes,
            "im1_rpdes": im1_rpdes,
            "im2_lpdes": im2_lpdes,
            "im2_rpdes": im2_rpdes,
            "im1_lpreddes": im1_lpreddes,
            "im1_rpreddes": im1_rpreddes,
            "im2_lpreddes": im2_lpreddes,
            "im2_rpreddes": im2_rpreddes,
        }

        return endpoint

    def prepare(self,batch):
        pass


    def inference(self, im_data, im_info, im_raw):
        # import time
        # start_time = time.time()
        # for i in range(100):
        #     im_rawsc, im_scale, im_orint = self.det(im_data)
        # end_time = time.time()
        # print((end_time-start_time)/100)
        im_rawsc, im_scale, im_orint = self.det(im_data)
        im_score = self.det.process(im_rawsc)[0]
        im_topk = topk_map(im_score, self.TOPK)
        kpts = im_topk.nonzero()  # (B*topk, 4)
        cos, sim = im_orint.squeeze().chunk(chunks=2, dim=-1)
        cos = cos.masked_select(im_topk.to(torch.bool))  # (B*topk)
        sim = sim.masked_select(im_topk.to(torch.bool))  # (B*topk)
        im_orint = torch.cat((cos.unsqueeze(-1), sim.unsqueeze(-1)), dim=-1)
        im_patches = clip_patch(
            kpts,
            im_scale.masked_select(im_topk.to(torch.bool)),
            im_orint,
            im_info,
            im_raw,
            PSIZE=self.PSIZE,
        )  # (numkp, 1, 32, 32)
        # start_time = time.time()
        # for i in range(100):
        #     im_des = self.des(im_patches)
        # end_time = time.time()
        # print((end_time - start_time) / 100)
        im_des = self.des(im_patches)
        # import pdb
        # pdb.set_trace()
        # from matplotlib import pyplot as plt
        # plt.imshow(scoremap_gaus[0, , :, :].cpu().detach().numpy())
        returnscale = im_scale.masked_select(im_topk.to(torch.bool))
        returnscore = im_rawsc.masked_select(im_topk.to(torch.bool))
        return im_scale, kpts, im_des, returnscale, im_orint, returnscore

    def gtscore(self, right_score, homolr):
        im2_score = right_score
        im2_score = filter_border(im2_score)

        # warp im2_score to im1w_score and calculate visible_mask
        im1w_score = warp(im2_score, homolr)
        im1visible_mask = warp(
            im2_score.new_full(im2_score.size(), fill_value=1, requires_grad=True),
            homolr,
        )

        im1gt_score, topk_mask, topk_value = self.det.process(im1w_score)

        return im1gt_score, topk_mask, topk_value, im1visible_mask

    @staticmethod
    def gt_scale_orin(im2_scale, im2_orin, homo12, homo21):
        B, H, W, C = im2_scale.size()
        im2_cos, im2_sin = im2_orin.squeeze().chunk(chunks=2, dim=-1)  # (B, H, W, 1)
        # im2_tan = im2_sin / im2_cos

        # each centX, centY, centZ is (B, H, W, 1)
        centX, centY, centZ = imgBatchXYZ(B, H, W).to(im2_scale.device).chunk(3, dim=3)

        """get im1w scale maps"""
        half_scale = im2_scale // 2
        centXYZ = torch.cat((centX, centY, centZ), dim=3)  # (B, H, W, 3)
        upXYZ = torch.cat((centX, centY - half_scale, centZ), dim=3)
        bottomXYZ = torch.cat((centX, centY + half_scale, centZ), dim=3)
        rightXYZ = torch.cat((centX + half_scale, centY, centZ), dim=3)
        leftXYZ = torch.cat((centX - half_scale, centY, centZ), dim=3)

        centXYw = transXYZ_2_to_1(centXYZ, homo21)  # (B, H, W, 2) (x, y)
        centXw, centYw = centXYw.chunk(chunks=2, dim=-1)  # (B, H, W, 1)
        centXYw = centXYw.long()
        upXYw = transXYZ_2_to_1(upXYZ, homo21).long()
        rightXYw = transXYZ_2_to_1(rightXYZ, homo21).long()
        bottomXYw = transXYZ_2_to_1(bottomXYZ, homo21).long()
        leftXYw = transXYZ_2_to_1(leftXYZ, homo21).long()

        upScale = MSD(upXYw, centXYw)
        rightScale = MSD(rightXYw, centXYw)
        bottomScale = MSD(bottomXYw, centXYw)
        leftScale = MSD(leftXYw, centXYw)
        centScale = (upScale + rightScale + bottomScale + leftScale) / 4  # (B, H， W, 1)

        """get im1w orintation maps"""
        offset_x, offset_y = im2_scale * im2_cos, im2_scale * im2_sin  # (B, H, W, 1)
        offsetXYZ = torch.cat((centX + offset_x, centY + offset_y, centZ), dim=3)
        offsetXYw = transXYZ_2_to_1(offsetXYZ, homo21)  # (B, H, W, 2) (x, y)
        offsetXw, offsetYw = offsetXYw.chunk(chunks=2, dim=-1)  # (B, H, W, 1)
        offset_ww, offset_hw = offsetXw - centXw, offsetYw - centYw  # (B, H, W, 1)
        offset_rw = (offset_ww ** 2 + offset_hw ** 2 + 1e-8).sqrt()
        # tan = offset_hw / (offset_ww + 1e-8)  # (B, H, W, 1)
        cos_w = offset_ww / (offset_rw + 1e-8)  # (B, H, W, 1)
        sin_w = offset_hw / (offset_rw + 1e-8)  # (B, H, W, 1)
        # atan_w = np.arctan(tan.cpu().detach())  # (B, H, W, 1)

        # get left scale by transXYZ_2_to_1
        map_xy_2_to_1 = transXYZ_2_to_1(centXYZ, homo12).round().long()  # (B, H, W, 2)
        x, y = map_xy_2_to_1.chunk(2, dim=3)  # each x and y is (B, H, W, 1)
        x = x.clamp(min=0, max=W - 1)
        y = y.clamp(min=0, max=H - 1)

        # (B, H, W, 1)
        im1w_scale = centScale[
            torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
        ].view(im2_scale.size())

        # (B, H, W, 1, 2)
        im1w_cos = cos_w[
            torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
        ].view(im2_cos.size())
        im1w_sin = sin_w[
            torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
        ].view(im2_sin.size())
        im1w_orin = torch.cat((im1w_cos, im1w_sin), dim=-1)
        im1w_orin = L2Norm(im1w_orin, dim=-1).to(im2_orin.device)
        im1w_orin = im1w_orin.unsqueeze(0)
        return im1w_scale, im1w_orin

    def criterion(self, endpoint):

        im1_score = endpoint["im1_score"]
        im1_gtsc = endpoint["im1_gtsc"]
        im1_visible = endpoint["im1_visible"]

        im2_score = endpoint["im2_score"]
        im2_gtsc = endpoint["im2_gtsc"]
        im2_visible = endpoint["im2_visible"]

        im1_limc = endpoint["im1_limc"]
        im1_rimcw = endpoint["im1_rimcw"]
        im2_limc = endpoint["im2_limc"]
        im2_rimcw = endpoint["im2_rimcw"]

        im1_lpdes = endpoint["im1_lpdes"]
        im1_rpdes = endpoint["im1_rpdes"]
        im2_lpdes = endpoint["im2_lpdes"]
        im2_rpdes = endpoint["im2_rpdes"]

        im1_lpreddes = endpoint["im1_lpreddes"]
        im1_rpreddes = endpoint["im1_rpreddes"]
        im2_lpreddes = endpoint["im2_lpreddes"]
        im2_rpreddes = endpoint["im2_rpreddes"]
        #
        # score loss
        #
        im1_scloss = self.det.loss(im1_score, im1_gtsc, im1_visible)
        im2_scloss = self.det.loss(im2_score, im2_gtsc, im2_visible)
        score_loss = (im1_scloss + im2_scloss) / 2.0 * self.SCORE_W

        #
        # pair loss
        #
        im1_pairloss = distance_matrix_vector(im1_lpreddes, im1_rpreddes).diag().mean()
        im2_pairloss = distance_matrix_vector(im2_lpreddes, im2_rpreddes).diag().mean()
        pair_loss = (im1_pairloss + im2_pairloss) / 2.0 * self.PAIR_W

        #
        # hard loss
        #
        im1_hardloss = self.des.loss(im1_lpdes, im1_rpdes, im1_limc, im1_rimcw)
        im2_hardloss = self.des.loss(im2_lpdes, im2_rpdes, im2_limc, im2_rimcw)
        hard_loss = (im1_hardloss + im2_hardloss) / 2.0

        # loss summary
        det_loss = score_loss + pair_loss
        des_loss = hard_loss

        PLT_SCALAR = {}
        PLT = {"scalar": PLT_SCALAR}

        PLT_SCALAR["score_loss"] = score_loss
        PLT_SCALAR["pair_loss"] = pair_loss
        PLT_SCALAR["hard_loss"] = hard_loss

        # import pdb
        # pdb.set_trace()
        return PLT, det_loss.mean(), des_loss.mean()

    def detectAndCompute(self, im_path, device, output_size):
        """
        detect keypoints and compute its descriptor
        :param im_path: image path
        :param device: cuda or cpu
        :param output_size: resacle size
        :return: kp (#keypoints, 4) des (#keypoints, 128)
        """
        import numpy as np
        from skimage import io, color
        from utils.image_utils import im_rescale

        img = io.imread(im_path)

        # Gray
        # img_raw = img = img/255
        img_raw = img = np.expand_dims(color.rgb2gray(img), -1)
        # Rescale
        # output_size = (240, 320)
        img, _, _, sw, sh = im_rescale(img, output_size)
        img_info = np.array([sh, sw])

        # to tensor
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = torch.from_numpy(img.transpose((2, 0, 1)))[None, :].to(
            device, dtype=torch.float
        )
        img_info = torch.from_numpy(img_info)[None, :].to(device, dtype=torch.float)
        img_raw = torch.from_numpy(img_raw.transpose((2, 0, 1)))[None, :].to(
            device, dtype=torch.float
        )

        # inference
        _, kp, des, scale, angle, score = self.inference(img, img_info, img_raw)

        return kp, des, img, scale, angle, score