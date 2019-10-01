import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_receptive_field import receptive_field, receptive_field_for_unit

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # vgg16
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

        # 1 kong
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=2,
        )

        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

receptive_field_dict = receptive_field(model, (1, 256, 256))
# receptive_field_for_unit(receptive_field_dict, "2", (2,2))a