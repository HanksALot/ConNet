import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from .basic_conv_block import BasicConvBlockEnc
from .basic_conv_block import BasicConvBlockDec
from .up_conv_block import UpConvBlock


class Linknet3D2D(BaseModule):
    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=None,
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None,
                 temporal_kernel=3,
                 tag_3d2d=True):
        channel_list = [base_channels * (2 ** x) for x in range(num_stages)]

        assert isinstance(conv_cfg, list) or conv_cfg is None, "conv_cfg must be a list or None"
        assert isinstance(norm_cfg, list) or norm_cfg is None, "norm_cfg must be a list or None"

        if conv_cfg is None:
            conv_cfg = [{'type': 'Conv3d'}, None]
        if norm_cfg is None:
            norm_cfg = [dict(type='BN3d', requires_grad=True),
                        dict(type='BN', requires_grad=True)]
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        if upsample_cfg is None:
            upsample_cfg = dict(type='InterpConv')
        super(Linknet3D2D, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, ' \
            f'while the strides is {strides}, the length of ' \
            f'strides is {len(strides)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, ' \
            f'while the enc_num_convs is {enc_num_convs}, the length of ' \
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages - 1), \
            'The length of dec_num_convs should be equal to (num_stages-1), ' \
            f'while the dec_num_convs is {dec_num_convs}, the length of ' \
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(downsamples) == (num_stages - 1), \
            'The length of downsamples should be equal to (num_stages-1), ' \
            f'while the downsamples is {downsamples}, the length of ' \
            f'downsamples is {len(downsamples)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, ' \
            f'while the enc_dilations is {enc_dilations}, the length of ' \
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is ' \
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages - 1), \
            'The length of dec_dilations should be equal to (num_stages-1), ' \
            f'while the dec_dilations is {dec_dilations}, the length of ' \
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is ' \
            f'{num_stages}.'
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels
        self.channel_list = channel_list
        self.temporal_kernel = temporal_kernel
        self.tag_3d2d = tag_3d2d

        self.enc_first = ConvModule(in_channels, base_channels, kernel_size=3, stride=1, padding=1,
                                    conv_cfg=conv_cfg[0], norm_cfg=norm_cfg[0], act_cfg=act_cfg)

        self.encoder = nn.ModuleList()
        if self.tag_3d2d is True:
            self.conv3d_2d = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            if i != 0:
                upsample = (strides[i] != 1 or downsamples[i - 1])
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlockDec,
                        in_channels=base_channels * 2 ** i,
                        skip_channels=base_channels * 2 ** (i - 1),
                        out_channels=base_channels * 2 ** (i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg[1],
                        norm_cfg=norm_cfg[1],
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None,
                        dcn=None,
                        plugins=None))

            enc_conv_block = []
            for j in range(0, enc_num_convs[i]):
                downsample = None
                if i != 0 and j == 0:
                    if conv_cfg[0] is None:
                        downsample = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels=base_channels * 2 ** i,
                                      kernel_size=1, stride=2),
                            nn.BatchNorm2d(base_channels * 2 ** i))
                    else:
                        downsample = nn.Sequential(
                            nn.Conv3d(in_channels, out_channels=base_channels * 2 ** i,
                                      kernel_size=1, stride=(1, 2, 2)),
                            nn.BatchNorm3d(base_channels * 2 ** i))

                if i == 0:
                    in_channels_bcb = base_channels
                    stride_bcb = 1
                elif i != 0 and j == 0:
                    in_channels_bcb = base_channels * 2 ** (i - 1)
                    stride_bcb = 2
                else:
                    in_channels_bcb = base_channels * 2 ** i
                    stride_bcb = 1
                enc_conv_block.append(
                    BasicConvBlockEnc(
                        in_channels=in_channels_bcb,
                        out_channels=base_channels * 2 ** i,
                        stride=stride_bcb,
                        downsample=downsample,
                        dilation=enc_dilations[i],
                        conv_cfg=conv_cfg[0],
                        norm_cfg=norm_cfg[0]))
            self.encoder.append((nn.Sequential(*enc_conv_block)))
            in_channels = base_channels * 2 ** i

            if self.tag_3d2d is True:
                self.conv3d_2d.append(
                    nn.Conv3d(in_channels, in_channels, kernel_size=(temporal_kernel, 1, 1),
                              stride=(1, 1, 1), padding=(0, 0, 0)))

    def forward(self, x):
        assert self.temporal_kernel == len(x)
        if len(x) == 1:
            x = x[0]
            self._check_input_divisible(x)
            x = self.enc_first(x)

            enc_outs = []
            for enc in self.encoder:
                x = enc(x)
                enc_outs.append(x)
        else:
            x = torch.stack(x, dim=2)
            self._check_input_divisible(x)
            x = self.enc_first(x)

            enc_outs_3d = []
            for enc in self.encoder:
                x = enc(x)
                enc_outs_3d.append(x)

            enc_outs = []
            for i in range(len(self.conv3d_2d)):
                x = self.conv3d_2d[i](enc_outs_3d[i])
                b, c, t, h, w = x.shape
                x = x.view(b, c, h, w)
                enc_outs.append(x)

        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(Linknet3D2D, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) and (
                w % whole_downsample_rate == 0), \
            f'The input image size {(h, w)} should be divisible by the whole ' \
            f'downsample rate {whole_downsample_rate}, when num_stages is ' \
            f'{self.num_stages}, strides is {self.strides}, and downsamples ' \
            f'is {self.downsamples}.'
