import torch
import torch.nn as nn
from mmseg.models.builder import BACKBONES
from .encoder_conv3d_block import ResConv3DBlock, DownConv3DBlock
from .dac_block import DACBlock
from .decoder_conv2d_block import UpsampleBlock, Conv2Block1, Conv2Block2


@BACKBONES.register_module()
class Semi3dTeNet(nn.Module):
    def __init__(self, n_ch=3):
        super(Semi3dTeNet, self).__init__()

        self.conv0 = nn.Conv3d(n_ch, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        self.conv1 = ResConv3DBlock(64)
        self.conv1_1 = DownConv3DBlock(64, 64)
        self.conv1_3d_2d = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.conv2 = ResConv3DBlock(64)
        self.conv2_1 = DownConv3DBlock(64, 64)
        self.conv2_3d_2d = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.conv3 = ResConv3DBlock(64)
        self.conv3_1 = DownConv3DBlock(64, 128)
        self.conv3_3d_2d = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.conv4 = ResConv3DBlock(128)
        self.conv4_1 = DownConv3DBlock(128, 256)
        self.conv4_3d_2d = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.conv5 = ResConv3DBlock(256)
        self.conv5_1 = DownConv3DBlock(256, 512)
        self.conv5_3d_2d = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.dac = DACBlock(512)

        self.up1 = UpsampleBlock(512, 256)
        self.up1_1 = Conv2Block1(256)

        self.up2 = UpsampleBlock(256, 128)
        self.up2_1 = Conv2Block1(128)

        self.up3 = UpsampleBlock(128, 64)
        self.up3_1 = Conv2Block1(64)

        self.up4 = UpsampleBlock(64, 64)
        self.up4_1 = Conv2Block1(64)

        self.up5 = UpsampleBlock(64, 64)
        self.up5_1 = Conv2Block2(64)

    def forward(self, x):
        # frames shape (B, C, T, H, W), img and gt shape (B, C, H, W)
        x = torch.stack(x, dim=2)
        conv0 = self.conv0(x)

        conv1 = self.conv1(conv0)
        conv1_1 = self.conv1_1(conv1)
        conv1_3d_2d = self.conv1_3d_2d(conv1_1)
        conv1_3d_2d = conv1_3d_2d.view(-1, 64, 256, 256)

        conv2 = self.conv2(conv1_1)
        conv2_1 = self.conv2_1(conv2)
        conv2_3d_2d = self.conv2_3d_2d(conv2_1)
        conv2_3d_2d = conv2_3d_2d.view(-1, 64, 128, 128)

        conv3 = self.conv3(conv2_1)
        conv3_1 = self.conv3_1(conv3)
        conv3_3d_2d = self.conv3_3d_2d(conv3_1)
        conv3_3d_2d = conv3_3d_2d.view(-1, 128, 64, 64)

        conv4 = self.conv4(conv3_1)
        conv4_1 = self.conv4_1(conv4)
        conv4_3d_2d = self.conv4_3d_2d(conv4_1)
        conv4_3d_2d = conv4_3d_2d.view(-1, 256, 32, 32)

        conv5 = self.conv5(conv4_1)
        conv5_1 = self.conv5_1(conv5)
        conv5_3d_2d = self.conv5_3d_2d(conv5_1)
        conv5_3d_2d = conv5_3d_2d.view(-1, 512, 16, 16)

        conv6_dac = self.dac(conv5_3d_2d)

        up1 = self.up1(conv6_dac)
        up1_1 = self.up1_1(conv4_3d_2d, up1)

        up2 = self.up2(up1_1)
        up2_1 = self.up2_1(conv3_3d_2d, up2)

        up3 = self.up3(up2_1)
        up3_1 = self.up3_1(conv2_3d_2d, up3)

        up4 = self.up4(up3_1)
        up4_1 = self.up4_1(conv1_3d_2d, up4)

        up5 = self.up5(up4_1)
        out = [up2_1, up3_1, up4_1, up5]

        return out
