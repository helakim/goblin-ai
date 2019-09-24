# *********************************************************************
# @Project    image_classification
# @FILE       convolution
# @Copyright: kyung_tae_kim (hela.kim) 
# @Created:   19. 9. 24.
#
#            7''  Q..\
#         7         (
#       7  /    _q.  /
#     _7 . ___  /VVvv-'_                                            .
#    7/ / /~- \_\\      '-._     .-'                      /       //
#   ./ ( /--/||'=.__  '::. '-'' {             ___   /  //     ./{
#  V   V-~-| ||   __''_   ':::.   ''-~.___.-'' /  // / {   /  {  /
#   VV/---|/ \ .'__'. '.    '::                     _        ''.
#   / /~~~~||VVV/ /  \ )  \        _ __ ___   ___ ___(_) | | __ _   .::'
#  / (--~\\.-' /    \'   \::::. | '_  _ \ / _ \_  / | | |/ _ | :::'
# /..\    /..\__/      '     '::: | | | | | | (_) / /| | | | (_| | ::
# vVVv    vVVv                 ': |_| |_| |_|\___/___|_|_|_|\__,_| ''
#
# *********************************************************************
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ops.activation import SigmoidWeightedLinearUnits
from models.squeeze import SeExcitation

class ResidualLinearBottlenecks(nn.Module):

    def __init__(self, in_place, out_place, kernel_size, stride, expand_rate=1.0, squeeze_aspect_ratio=0.25, drop_connect_aspect_rate=0.2):
        """
        Args:
            :param in_place:
            :param out_place:
            :param kernel_size:
            :param stride:
            :param expand_rate:
            :param squeeze_aspect_ratio:
            :param drop_connect_aspect_rate:
        """
        super(ResidualLinearBottlenecks, self).__init__()
        # TODO: add assert code [level: 1]
        self.in_place = in_place
        self.out_place = out_place
        self.kernel_size = kernel_size
        self.expand_rate = expand_rate
        self.stride = stride
        self.squeeze_aspect_ratio = squeeze_aspect_ratio
        self.eps = 1e-3
        # ------
        # local variables for expand convolution units
        expand_units_channels = np.int(self.in_place * self.expand_rate)
        squeeze_units_channels = np.max(1, np.int(self.in_place * self.squeeze_aspect_ratio))
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(self.in_place, out_channels=self.out_place, kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(expand_units_channels, momentum=0.01, eps=self.eps),
                SigmoidWeightedLinearUnits()
            )
            self.in_place = expand_units_channels
        # ------
        # DepthIse Convolution Layers
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.in_place, expand_units_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.kernel_size // 2, groups=expand_units_channels, bias=False),
            nn.BatchNorm2d(expand_units_channels, momentum=0.01, eps=self.eps),
            SigmoidWeightedLinearUnits()
        )
        # ------
        # Squeeze Excitation Layers
        self.squeeze_excitation = SeExcitation(expand_units_channels, squeeze_units_channels)
        # ------
        # Projection Convolution Layers
        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_units_channels, out_channels=self.in_place, kernel_size=1, stride=1, padding=0.1, bias=False),
            nn.BatchNorm2d(self.in_place, momentum=0.01, eps=self.eps)
        )
        self.with_skip = self.stride == 1
        self.drop_connect_aspect_rate = torch.tensor(drop_connect_aspect_rate, requires_grad=False)

    def __drop_connection_func(self, x):
        """
            :param x: 2-dimensional tensor
        """
        drop_keep_prob = 1.0 - self.drop_connect_aspect_rate
        drop_mask_scale = torch.randn(x.shape[0], 1, 1, 1) + drop_keep_prob
        drop_mask_scale = drop_mask_scale.type_as(x)
        drop_mask_scale.floor_()
        ops_results = drop_mask_scale * x / drop_keep_prob

        return ops_results

    def forward(self, input_x):
        z_vector = input_x
        if self.expansion_conv is not None:
            x = self.expansion_conv(input_x)
        x = self.project_conv(self.squeeze_excitation(self.depthwise_conv(input_x)))
        # -----
        # for like identity mapping (search for residual)
        if input_x.shape == z_vector.shape and self.with_skip:
            # skip-connection
            if self.training and self.drop_connect_aspect_rate is not None:
                self.__drop_connection_func(input_x)
            input_x += z_vector

        return input_x