import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, conv_cfg=None):
    """3x3 convolution with padding"""
    if conv_cfg is None:
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)
    else:
        return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=(1, stride, stride),
                         padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicConvBlockEnc(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, conv_cfg=None, norm_cfg=None):
        super(BasicConvBlockEnc, self).__init__()
        if norm_cfg['type'] == 'BN3d':
            norm_layer = nn.BatchNorm3d
        else:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride, conv_cfg=conv_cfg)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, conv_cfg=conv_cfg)
        self.bn2 = norm_layer(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicConvBlockDec(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 dcn=None,
                 plugins=None):
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        super(BasicConvBlockDec, self).__init__()

        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out
