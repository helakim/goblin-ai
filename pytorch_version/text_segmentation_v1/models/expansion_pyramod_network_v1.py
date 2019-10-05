# *********************************************************************
# @Project    text_segmentation_v1
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   03/10/2019
#
#            7''  Q..\
#         _7         (_
#       _7  _/    _q.  /
#     _7 . ___  /VVvv-'_                                            .
#    7/ / /~- \_\\      '-._     .-'                      /       //
#   ./ ( /-~-/||'=.__  '::. '-~'' {             ___   /  //     ./{
#  V   V-~-~| ||   __''_   ':::.   ''~-~.___.-'' _/  // / {_   /  {  /
#   VV/-~-~-|/ \ .'__'. '.    '::                     _ _ _        ''.
#   / /~~~~||VVV/ /  \ )  \        _ __ ___   ___ ___(_) | | __ _   .::'
#  / (~-~-~\\.-' /    \'   \::::. | '_ ` _ \ / _ \_  / | | |/ _` | :::'
# /..\    /..\__/      '     '::: | | | | | | (_) / /| | | | (_| | ::
# vVVv    vVVv                 ': |_| |_| |_|\___/___|_|_|_|\__,_| ''
#
# *********************************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone_v1 import custom_deep_residual_50


def conv_3x3(in_channels, out_channels, stride=1):
    units = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,
                      bias=False, groups=1)

    return units


def _up_sample(x, y, scale=1):
    # _, _, height, width = y.size()
    height, width = y.size()[2:]
    return F.interpolate(input=x, size=(height // scale, width // scale), mode='bilinear', align_corners=True)


class AttentionBlocks(nn.Module):

    def __init__(self, in_channels, out_channels=512):
        """
        Description:
            - The pyramid attention modules fuses features from under three different pyramid scale by
            implementing a U-NET(`Convolution Networks for Biomedical Image Segmentation`) shape
            structure like Feature Pyramid Networks, To better extract context from different pyramid
            scales, we use (3x3, 5x5, 7x7) convolution in pyramid structure respectively since the
            resolution of high-level feature maps is small, using large kernel size does not bring
            too much computation burden.

        Args:
            :param in_channels: in put convolution planes
            :param out_channels: out put convolution planes
        """
        super(AttentionBlocks, self).__init__()
        # ------
        # first branch
        self.in_channels = in_channels
        self.c_master = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.b_master = nn.BatchNorm2d(in_channels)

        # ------
        # global pooling convolution branch
        self.c_global_convolution = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.b_global_convolution = nn.BatchNorm2d(in_channels)
        self.act_global_convolution = nn.ReLU(inplace=True)

        # ------
        # dilated convolution branch with local convolution (1x1)
        self.c_dilate_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=3, bias=False)
        self.b_dilate_1 = nn.BatchNorm2d(out_channels)

        self.c_dilate_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=6, bias=False)
        self.b_dilate_2 = nn.BatchNorm2d(out_channels)

        self.c_dilate_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=12, bias=False)
        self.b_dilate_3 = nn.BatchNorm2d(out_channels)
        self.c_local_1x1 = nn.Conv2d(3 * out_channels, in_channels, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            :param x: input_shape [batch, channel(2048), height, width]
        """
        # ------
        # first branch
        x_master = self.b_master(self.c_master(x))

        # ------
        # global pooling convolution branch
        x_global_convolution = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.in_channels, 1, 1)
        x_global_convolution = self.act_global_convolution(
            self.b_global_convolution(self.c_global_convolution(x_global_convolution)))

        # ------
        # dilated convolution branch with local convolution (1x1)
        dilate_1 = self.b_dilate_1(self.c_dilate_1(x))
        dilate_2 = self.b_dilate_2(self.c_dilate_2(x))
        dilate_3 = self.b_dilate_3(self.c_dilate_3(x))

        # ------
        # up sampling
        dilate_1 = _up_sample(dilate_1, x_master)
        dilate_2 = _up_sample(dilate_2, x_master)
        dilate_3 = _up_sample(dilate_3, x_master)

        x_global_convolution = _up_sample(x_global_convolution, x_master)

        y = torch.cat((dilate_3, dilate_2, dilate_1), dim=1)
        y = self.c_local_1x1(y)
        y = y * x_master
        y = self.relu(y + x_global_convolution)  # feature map size is `1024`

        return y


class GlobalAttentionUP(nn.Module):

    def __init__(self, channels_high, channel_low):
        """
            :param channels_high: height of convolution channels
            :param channel_low: low of convolution channels
        """
        super(GlobalAttentionUP, self).__init__()
        self.c_3x3 = nn.Conv2d(channel_low, channel_low, kernel_size=3, padding=0, bias=False)
        self.b_low = nn.BatchNorm2d(channel_low)

        self.c_1x1 = nn.Conv2d(channels_high, channel_low, kernel_size=1, padding=0, bias=False)
        self.b_high = nn.BatchNorm2d(channel_low)

        self.act = nn.ReLU(inplace=True)

    def forward(self, feature_maps_high, feature_maps_low):
        batch, channel, height, width = feature_maps_high.sahpe

        # -------
        # high level features
        feature_maps_high_gp = nn.AvgPool2d(feature_maps_high.shape[2:])(feature_maps_high).view(len(feature_maps_high), channel, 1, 1)
        feature_maps_high_gp = self.relu(self.b_high(self.c_1x1(feature_maps_high_gp)))

        # ------
        # low level features with mask
        feature_maps_low_mask = self.b_low(self.c_3x3(feature_maps_low))
        feature_maps_attention = feature_maps_low_mask * feature_maps_high_gp

        # -------
        # up-samplings
        feature_maps_high_up = _up_sample(feature_maps_high, feature_maps_low_mask)
        feature_maps_high_up = self.c_1x1(feature_maps_high_up)

        y = self.act(feature_maps_high_up * feature_maps_attention)  # same feature size as passed residual stage

        return y


class ExpansionPyramidNetwork(nn.Module):

    def __init__(self, result_num, scale=1):
        """
        Args:
            :param backbone:
            :param result_num: fully convolution channels for regression
            :param scale: image factor scale
        """
        super(ExpansionPyramidNetwork, self).__init__()
        self.scale = scale
        self.backbone = custom_deep_residual_50()
        # ------
        # first top layer + reduce convolution channels
        self.toplayer = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0)
        # ------
        # second middle layer (stacked 3 convolution layer)
        self.smooth1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # ------
        # skip connection layer [stacked 3 convolution layer] -> 1 x 1 convolution
        self.latlayer1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)

        # -------
        # prepare high resolution
        self.latlayer5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.latlayer6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(nn.Conv2d(in_channels=1024,
                                            out_channels=256,
                                            kernel_size=3,
                                            padding=1,
                                            stride=1),
                                  nn.BatchNorm2d(num_features=256),
                                  nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(in_channels=256, out_channels=result_num, kernel_size=1, stride=1)

    def forward(self, x):
        _, _, height, width = x.size()
        c_1, c_2, c_3, c_4, c_5 = self.backbone(x)
        # ------
        # like feature pyramid network
        # https://arxiv.org/abs/1612.03144
        # top-down pathway -> low-solution (semantically strong feature)
        pyramid_level_5 = self.toplayer(c_5)
        pyramid_level_4 = self._upsample_add(pyramid_level_5, self.latlayer1(c_4))
        pyramid_level_3 = self._upsample_add(pyramid_level_4, self.latlayer2(c_3))
        pyramid_level_2 = self._upsample_add(pyramid_level_3, self.latlayer3(c_2))
        # ------
        # middle layer
        pyramid_level_4 = self.smooth1(pyramid_level_4)
        pyramid_level_3 = self.smooth2(pyramid_level_3)
        pyramid_level_2 = self.smooth3(pyramid_level_2)
        # ------
        # feature extraction
        feature_extraction = self._upsample_cat(pyramid_level_2, pyramid_level_3, pyramid_level_4, pyramid_level_5)
        feature_extraction = self.conv(feature_extraction)
        feature_extraction = self.out_conv(feature_extraction)

        feature_extraction = F.interpolate(feature_extraction, size=(height // self.scale, width // self.scale),
                                           mode='bilinear', align_corners=True)

        return feature_extraction

    @staticmethod
    def _upsample_add(x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True) + y

    @staticmethod
    def _upsample_cat(pyramid_2, pyramid_3, pyramid_4, pyramid_5):
        height, width = pyramid_2.size()[2:]
        pyramid_3 = F.interpolate(pyramid_3, size=(height, width), mode='bilinear', align_corners=True)
        pyramid_4 = F.interpolate(pyramid_4, size=(height, width), mode='bilinear', align_corners=True)
        pyramid_5 = F.interpolate(pyramid_5, size=(height, width), mode='bilinear', align_corners=True)

        return torch.cat([pyramid_2, pyramid_3, pyramid_4, pyramid_5], dim=1)