# *********************************************************************
# @Project    goblin-ai
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   30/09/2019
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


def local_convolution(in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=1):
    unit_x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)

    return unit_x


class FeatureEnhanceModule(nn.Module):

    def __init__(self, channel_size):
        """
        Description:
            - Feature Enhance Module is able to convolution enhance original features map to make them more discriminable and robust
        Args:
            :param channel_size: size of convolution channels

        Testing: True
        """
        super(FeatureEnhanceModule, self).__init__()
        filter_size = [128, 256]
        self.channel_size = channel_size
        # ------
        # feature extraction
        self.cpm1 = local_convolution(in_channels=self.channel_size, out_channels=filter_size[1])
        self.cpm2 = nn.Conv2d(in_channels=self.channel_size, out_channels=filter_size[1], kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = local_convolution(in_channels=filter_size[1], out_channels=filter_size[0])
        self.cpm4 = nn.Conv2d(in_channels=filter_size[1], out_channels=filter_size[0], kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = local_convolution(in_channels=filter_size[0], out_channels=filter_size[0])
        # self.cpm5 = local_convolution(in_channels=filter_size[1] / 2, out_channels=filter_size[1] / 2)

    def forward(self, x):
        convolution_1_1 = F.relu(self.cpm1(x), inplace=True)
        convolution_1_2 = F.relu(self.cpm2(x), inplace=True)

        convolution_2_1 = F.relu(self.cpm3(convolution_1_2), inplace=True)
        convolution_2_2 = F.relu(self.cpm4(convolution_1_2), inplace=True)

        convolution_3_1 = F.relu(self.cpm5(convolution_2_2), inplace=True)

        stacked_convolution_layer = torch.cat((convolution_1_1, convolution_2_1, convolution_3_1), dim=1)

        return stacked_convolution_layer