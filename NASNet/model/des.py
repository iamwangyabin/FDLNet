# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from utils.math_utils import distance_matrix_vector, pairwise_distances
from model.operations import IRFBlock,Identity

class HardNetNeiMask(nn.Module):
    def __init__(self, MARGIN, C):
        super(HardNetNeiMask, self).__init__()
        self.MARGIN = MARGIN
        self.C = C
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False),
            #######################################################################################################
            ## Stage 1: Identity(32,32,2)
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2, padding=0, groups=1, bias=not 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ## Stage 2: Identity(32,32,1)
            # pass
            ## Stage 3: Identity(32,64,2)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0, groups=1, bias=not 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ## Stage 4: IRF(64,64,exp = 1,stride = 1,ksize = 5)
            IRFBlock(64, 64, 1, 1, kernel=5),
            ## Stage 5:
            IRFBlock(64, 128, 3, 2, kernel=3),
            ## Stage 6:
            IRFBlock(128, 128, 1, 1, kernel=5, shuffle_type="mid", pw_group=2),
            #######################################################################################################
            nn.Conv2d(128, 128, kernel_size=4, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

        for p in self.features.parameters():
            p.requires_grad = True

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

    def loss(self, anchor, positive, anchor_kp, positive_kp):
        """
        HardNetNeiMask
        margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
        if set C=0 the loss function is same as hard loss.
        """
        "Input sizes between positive and negative must be equal."
        assert anchor.size() == positive.size()
        "Inputd must be a 2D matrix."
        assert anchor.dim() == 2

        dist_matrix = distance_matrix_vector(anchor, positive)
        eye = torch.eye(dist_matrix.size(1)).to(dist_matrix.device)

        # steps to filter out same patches that occur in distance matrix as negatives
        pos = dist_matrix.diag()
        dist_without_min_on_diag = dist_matrix + eye * 10

        # neighbor mask
        coo_dist_matrix = pairwise_distances(
            anchor_kp[:, 1:3].to(torch.float), anchor_kp[:, 1:3].to(torch.float)
        ).lt(self.C)
        dist_without_min_on_diag = (
            dist_without_min_on_diag + coo_dist_matrix.to(torch.float) * 10
        )
        coo_dist_matrix = pairwise_distances(
            positive_kp[:, 1:3].to(torch.float), positive_kp[:, 1:3].to(torch.float)
        ).lt(self.C)
        dist_without_min_on_diag = (
            dist_without_min_on_diag + coo_dist_matrix.to(torch.float) * 10
        )
        col_min = dist_without_min_on_diag.min(dim=1)[0]
        row_min = dist_without_min_on_diag.min(dim=0)[0]
        col_row_min = torch.min(col_min, row_min)

        # triplet loss
        hard_loss = torch.clamp(self.MARGIN + pos - col_row_min, min=0.0)
        hard_loss = hard_loss.mean()

        return hard_loss

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            try:
                nn.init.constant(m.bias.data, 0.01)
            except:
                pass
