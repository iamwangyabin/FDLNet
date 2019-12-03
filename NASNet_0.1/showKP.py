import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

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
)
des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
model = Network(
    det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
)

print(f"{gct()} : to device")
device = torch.device("cuda")
model = model.to(device)
resume = '/home/wang/workspace/Faster-net2/runs/03/model/e082_NN_0.165_NNT_0.397_NNDR_0.673_MeanMS_0.412.pth.tar'
print(f"{gct()} : in {resume}")
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint["state_dict"])

def to_cv2_kp(kp, scale, angle):
    # kp is like [batch_idx, y, x, channel]
    return cv2.KeyPoint(kp[2], kp[1], scale, angle)

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

kp1, des1, img1, scale1, angle1 = model.detectAndCompute('/home/wang/workspace/Faster-net2/ScalableNet_Net0.3/material/img3.png', device, (460, 600))
angle = np.degrees(np.arctan((angle1[:,1]/angle1[:,0]).detach().cpu().numpy()))
img1 = reverse_img(img1)
keypoints1 = list(map(to_cv2_kp, kp1, scale1, angle))
img = cv2.drawKeypoints(img1, keypoints1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('rfnet_keypoints_text.jpg', img)