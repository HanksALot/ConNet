import torch
import torch.nn as nn
from mmcv.cnn.bricks import build_conv_layer

from .linknet3d_block import Linknet3D2D


class IterEnd(nn.Module):
    def __init__(self, tag_3d2d=True, tag_fusion=True, **kwargs):
        super().__init__()
        self.tag_3d2d = tag_3d2d
        self.tag_fusion = tag_fusion

        unet_base = Linknet3D2D(tag_3d2d=self.tag_3d2d, **kwargs)

        if self.tag_fusion is True:
            channel_base = unet_base.channel_list[0]
            self.iter_a_chgch = build_conv_layer(
                cfg=kwargs['conv_cfg'][0], in_channels=channel_base * 2,
                out_channels=channel_base, kernel_size=3, stride=1, padding=1)

        self.base_first = unet_base.enc_first

        self.base_enc_0 = unet_base.encoder[0]
        self.base_enc_1 = unet_base.encoder[1]
        self.base_enc_2 = unet_base.encoder[2]
        self.base_enc_3 = unet_base.encoder[3]

        if self.tag_3d2d is True:
            self.base_3d_2d = unet_base.conv3d_2d

        self.base_dec_0 = unet_base.decoder[0]
        self.base_dec_1 = unet_base.decoder[1]
        self.base_dec_2 = unet_base.decoder[2]

    def forward(self, x, previous=None):
        if self.tag_fusion:
            x = self.base_first(x)
            x = torch.cat([x, previous], dim=1)
            b_x = self.iter_a_chgch(x)
        else:
            b_x = self.base_first(x)

        b_e0 = self.base_enc_0(b_x)
        b_e1 = self.base_enc_1(b_e0)
        b_e2 = self.base_enc_2(b_e1)
        b_n3 = self.base_enc_3(b_e2)

        if self.tag_3d2d is True:
            b_e0 = self.base_3d_2d[0](b_e0)
            b, c, t, h, w = b_e0.shape
            b_e0 = b_e0.view(b, c, h, w)
            b_e1 = self.base_3d_2d[1](b_e1)
            b, c, t, h, w = b_e1.shape
            b_e1 = b_e1.view(b, c, h, w)
            b_e2 = self.base_3d_2d[2](b_e2)
            b, c, t, h, w = b_e2.shape
            b_e2 = b_e2.view(b, c, h, w)
            b_n3 = self.base_3d_2d[3](b_n3)
            b, c, t, h, w = b_n3.shape
            b_n3 = b_n3.view(b, c, h, w)

        b_d2 = self.base_dec_2(b_e2, b_n3)
        b_d1 = self.base_dec_1(b_e1, b_d2)
        b_d0 = self.base_dec_0(b_e0, b_d1)

        return b_d0


class IterNotEnd(nn.Module):
    def __init__(self, tag_3d2d=False, tag_fusion=False, **kwargs):
        super().__init__()
        self.tag_3d2d = tag_3d2d
        self.tag_fusion = tag_fusion
        kwargs['conv_cfg'] = [{'type': 'Conv3d'}, {'type': 'Conv3d'}]
        kwargs['norm_cfg'] = [dict(type='BN3d', requires_grad=True),
                              dict(type='BN3d', requires_grad=True)]
        kwargs['upsample_cfg'] = dict(type='InterpConv3d')

        unet_iter_a = Linknet3D2D(tag_3d2d=self.tag_3d2d, **kwargs)

        channel_base = unet_iter_a.channel_list[0]
        if self.tag_fusion is True:
            self.iter_a_chgch = build_conv_layer(
                cfg=kwargs['conv_cfg'][0], in_channels=channel_base * 2,
                out_channels=channel_base, kernel_size=3, stride=1, padding=1)
        self.temporal_kernel = kwargs['temporal_kernel']
        self.end3d2d = nn.Conv3d(channel_base, channel_base, kernel_size=(self.temporal_kernel, 1, 1),
                                 stride=(1, 1, 1), padding=(0, 0, 0))

        self.iter_a_first = unet_iter_a.enc_first

        self.iter_a_enc_0 = unet_iter_a.encoder[0]
        self.iter_a_enc_1 = unet_iter_a.encoder[1]
        self.iter_a_enc_2 = unet_iter_a.encoder[2]
        self.iter_a_enc_3 = unet_iter_a.encoder[3]

        self.iter_a_dec_0 = unet_iter_a.decoder[0]
        self.iter_a_dec_1 = unet_iter_a.decoder[1]
        self.iter_a_dec_2 = unet_iter_a.decoder[2]

    def forward(self, x, previous=None):
        if self.tag_fusion:
            x = self.iter_a_first(x)
            x = torch.cat([x, previous], dim=1)
            i_a_x = self.iter_a_chgch(x)
        else:
            i_a_x = self.iter_a_first(x)

        i_a_e0 = self.iter_a_enc_0(i_a_x)
        i_a_e1 = self.iter_a_enc_1(i_a_e0)
        i_a_e2 = self.iter_a_enc_2(i_a_e1)
        i_a_n3 = self.iter_a_enc_3(i_a_e2)

        i_a_d2 = self.iter_a_dec_2(i_a_e2, i_a_n3)
        i_a_d1 = self.iter_a_dec_1(i_a_e1, i_a_d2)
        i_a_d0 = self.iter_a_dec_0(i_a_e0, i_a_d1)

        i_a_d0_end2d = self.end3d2d(i_a_d0)
        b, c, t, h, w = i_a_d0_end2d.shape
        i_a_d0_end2d = i_a_d0_end2d.view(b, c, h, w)

        return i_a_d0, i_a_d0_end2d


class DCtlLinkNet3DIter(nn.Module):
    def __init__(self, num_iters, **kwargs):
        super().__init__()
        self.num_iters = num_iters

        if self.num_iters == 0:
            self.tag_3d2d = kwargs['conv_cfg'][0] == dict(type='Conv3d')
            self.iter_end = IterEnd(tag_3d2d=self.tag_3d2d, tag_fusion=False, **kwargs)
        elif self.num_iters == 1:
            self.iter_zero = IterNotEnd(tag_3d2d=False, tag_fusion=False, **kwargs)
            self.iter_end = IterEnd(tag_3d2d=True, tag_fusion=True, **kwargs)
        elif self.num_iters == 2:
            self.iter_zero = IterNotEnd(tag_3d2d=False, tag_fusion=False, **kwargs)
            self.iter_one = IterNotEnd(tag_3d2d=False, tag_fusion=True, **kwargs)
            self.iter_end = IterEnd(tag_3d2d=True, tag_fusion=True, **kwargs)
        elif self.num_iters == 3:
            self.iter_zero = IterNotEnd(tag_3d2d=False, tag_fusion=False, **kwargs)
            self.iter_one = IterNotEnd(tag_3d2d=False, tag_fusion=True, **kwargs)
            self.iter_two = IterNotEnd(tag_3d2d=False, tag_fusion=True, **kwargs)
            self.iter_end = IterEnd(tag_3d2d=True, tag_fusion=True, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.num_iters == 0:
            x = torch.stack(x, dim=2) if self.tag_3d2d else x[-1]
            outputs = self.iter_end(x)
            return outputs
        elif self.num_iters == 1:
            x = torch.stack(x, dim=2)
            zero_dec3d, zero_dec2d = self.iter_zero(x)
            outputs = [zero_dec2d]

            end_dec2d = self.iter_end(x, zero_dec3d)
            outputs.append(end_dec2d)

            outputs.reverse()
            return outputs
        elif self.num_iters == 2:
            x = torch.stack(x, dim=2)
            zero_dec3d, zero_dec2d = self.iter_zero(x)
            outputs = [zero_dec2d]

            one_dec3d, one_dec2d = self.iter_one(x, zero_dec3d)
            outputs.append(one_dec2d)

            end_dec2d = self.iter_end(x, one_dec3d)
            outputs.append(end_dec2d)

            outputs.reverse()
            return outputs
        elif self.num_iters == 3:
            x = torch.stack(x, dim=2)
            zero_dec3d, zero_dec2d = self.iter_zero(x)
            outputs = [zero_dec2d]

            one_dec3d, one_dec2d = self.iter_one(x, zero_dec3d)
            outputs.append(one_dec2d)

            two_dec3d, two_dec2d = self.iter_two(x, one_dec3d)
            outputs.append(two_dec2d)

            end_dec2d = self.iter_end(x, two_dec3d)
            outputs.append(end_dec2d)

            outputs.reverse()
            return outputs
        else:
            raise NotImplementedError
