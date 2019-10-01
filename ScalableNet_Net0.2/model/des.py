# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from utils.math_utils import distance_matrix_vector, pairwise_distances



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

class HardNetNeiMask(nn.Module):
    def __init__(self, MARGIN, C):
        super(HardNetNeiMask, self).__init__()

        self.MARGIN = MARGIN
        self.C = C

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InvertedResidual(32, 32, 1, 1),
            InvertedResidual(32, 64, 2, 2),
            InvertedResidual(64, 64, 1, 1),
            InvertedResidual(64, 128, 2, 2),
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
        # import pdb
        # pdb.set_trace()
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
