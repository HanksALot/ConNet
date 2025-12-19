from functools import reduce
from operator import __add__

import torch
import torch.nn as nn


class ResConv3DBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResConv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(num_filter, num_filter, kernel_size=(3, 3, 3),
                               stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.InstanceNorm3d(num_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(num_filter, num_filter, kernel_size=(3, 3, 3),
                               stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.InstanceNorm3d(num_filter)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.add(x, out)
        out = self.relu(out)
        return out


class DownConv3DBlock(nn.Module):
    def __init__(self, in_num_filter, num_filter):
        super(DownConv3DBlock, self).__init__()
        # Internal parameters used to reproduce Tensorflow "Same" padding.
        # For some reasons, padding dimensions are reversed wrt kernel sizes,
        # first comes width then height in the 2D case.

        self.conv = nn.Conv3d(in_num_filter, num_filter, kernel_size=(2, 2, 2),
                              stride=(1, 2, 2), padding=(0, 0, 0))

        kernel_sizes = (2, 1, 1)
        conv_padding = reduce(__add__,
                              [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_sizes[::-1]])
        self.pad = nn.ZeroPad2d(conv_padding)

        self.bn = nn.BatchNorm3d(num_filter)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.pad(out)
        out = self.bn(out)
        out = self.relu(out)

        return out
