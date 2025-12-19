import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    def __init__(self, in_num_filter, num_filter):
        super(UpsampleBlock, self).__init__()

        self.deconv = nn.ConvTranspose2d(in_num_filter, in_num_filter, kernel_size=4, stride=2, padding=1)

        self.conv = nn.Conv2d(in_num_filter, num_filter, kernel_size=(1, 1), stride=(1, 1))

        self.bn = nn.BatchNorm2d(num_filter)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.deconv(x)

        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


class Conv2Block1(nn.Module):
    def __init__(self, num_filter):
        super(Conv2Block1, self).__init__()
        self.conv1 = nn.Conv2d(num_filter * 2, num_filter, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(num_filter)

    def forward(self, skip, x):
        out = torch.cat([skip, x], dim=1)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)
        return out


class Conv2Block2(nn.Module):
    def __init__(self, num_filter):
        super(Conv2Block2, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(num_filter)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.add(x, out)
        out = self.relu(out)
        return out
