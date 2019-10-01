# -*- coding: utf-8 -*-

import cv2
import torch
import random
import argparse
import numpy as np

from utils.common_utils import gct
from utils.eval_utils import nearest_neighbor_distance_ratio_match
from model.des import HardNetNeiMask
from model.det import RFDet
from model.network import Network
from config import cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="example")
    parser.add_argument("--imgpath", default=None, type=str)  # image path
    parser.add_argument("--resume", default=None, type=str)  # model path
    args = parser.parse_args()

    print(f"{gct()} : start time")

    random.seed(cfg.PROJ.SEED)
    torch.manual_seed(cfg.PROJ.SEED)
    np.random.seed(cfg.PROJ.SEED)

    print(f"{gct()} : model init")
    det = RFDet(
        cfg.TRAIN.score_com_strength,
        cfg.TRAIN.scale_com_strength,
        cfg.TRAIN.NMS_THRESH,
        cfg.TRAIN.NMS_KSIZE,
        cfg.TRAIN.TOPK,
        cfg.MODEL.GAUSSIAN_KSIZE,
        cfg.MODEL.GAUSSIAN_SIGMA,
        cfg.MODEL.KSIZE,
        cfg.MODEL.padding,
        cfg.MODEL.dilation,
        cfg.MODEL.scale_list,
        True
    )
    des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
    model = Network(
        det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
    )

    print(f"{gct()} : to device")
    device = torch.device("cuda")
    model = model.to(device)
    resume = args.resume
    print(f"{gct()} : in {resume}")
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint["state_dict"])

    ###############################################################################
    # detect and compute
    ###############################################################################
    img1_path, img2_path = args.imgpath.split("@")
    kp1, des1, img1 = model.detectAndCompute(img1_path, device, (600, 460))
    kp2, des2, img2 = model.detectAndCompute(img2_path, device, (460, 600))

    predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(des1, des2, kp2, 0.9)
    idx = predict_label.nonzero().view(-1)
    mkp1 = kp1.index_select(dim=0, index=idx.long())  # predict match keypoints in I1
    mkp2 = nn_kp2.index_select(dim=0, index=idx.long())  # predict match keypoints in I2

    def to_cv2_kp(kp):
        # kp is like [batch_idx, y, x, channel]
        return cv2.KeyPoint(kp[2], kp[1], 0)

    def to_cv2_dmatch(m):
        return cv2.DMatch(m, m, m, m)

    def reverse_img(img):
        """
        reverse image from tensor to cv2 format
        :param img: tensor
        :return: RBG image
        """
        img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
        img = (img * 255).astype(np.uint8)  # change to opencv format
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # gray to rgb
        return img

    img1, img2 = reverse_img(img1), reverse_img(img2)
    keypoints1 = list(map(to_cv2_kp, kp1))
    keypoints2 = list(map(to_cv2_kp, kp2))
    DMatch = list(map(to_cv2_dmatch, np.arange(0, len(keypoints1))))
    # 提取并计算特征点
    bf = cv2.BFMatcher(cv2.NORM_L1)
    matches = bf.knnMatch(des1.cpu().detach().numpy(), trainDescriptors=des2.cpu().detach().numpy(), k=2)
    good = [m for (m, n) in matches if m.distance < 0.8 * n.distance]
    # pdb.set_trace()
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 30.0)
    matchesMask = mask.ravel().tolist()
    nonmatchesMask = [1 if (j == 0) else 0 for j in matchesMask]

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=2)  # draw only inliers

    draw_params2 = dict(matchColor=(0, 0, 255),  # draw matches in green color
                        singlePointColor=(0, 0, 255),
                        matchesMask=nonmatchesMask,
                        flags=2)  # draw only inliers
    outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good, None, **draw_params)
    outImg = cv2.drawMatches(outImg[:, :460, :], keypoints1, outImg[:, 460:, :], keypoints2, good, None, **draw_params2)

    cv2.imwrite("outImg1.png", outImg)