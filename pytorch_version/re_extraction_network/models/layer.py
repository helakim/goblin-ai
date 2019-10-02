# *********************************************************************
# @Project    goblin-ai
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   02/10/2019
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
import torch.nn as nn


def convolution_3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    x_uint = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, bias=False)

    return x_uint


def local_convolution(in_channels, out_channels, stride=1):
    x_unit = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)

    return x_unit


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # ------
        # 1.convolution with batch normalization
        self.conv1 = local_convolution(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)
        # ------
        # 2.convolution with batch normalization
        self.conv2 = convolution_3x3(in_channels=out_planes, out_channels=out_planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=out_planes)
        # ------
        # 3.convolution with batch normalization + activation function
        self.conv3 = local_convolution(out_planes, out_channels=out_planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(num_features=out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # ------
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual_units = x
        # ------
        # convolution_stage_1
        feature_map = self.relu(self.bn1(self.conv1(x)))
        # ------
        # convolution_stage_2
        feature_map = self.relu(self.bn2(self.conv2(feature_map)))
        # ------
        # convolution_stage_3
        feature_map = self.bn3(self.conv3(feature_map))
        # ------
        # skip connection stage with residual units
        if self.downsample is not None:
            residual_units = self.downsample(x)

        feature_map += residual_units
        feature_map = self.relu(feature_map)

        return feature_map
