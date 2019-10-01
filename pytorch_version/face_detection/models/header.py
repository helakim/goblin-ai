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
import torch.nn as nn
import torch.nn.functional as F


def local_convolution(in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=1):
    unit_x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, bias=True)

    return unit_x


class MultiBoxHeadModule(nn.Module):

    def __init__(self, in_channels, out_channels, middle_res_channel=256):
        """
        Args:
            :param in_channels: input convolution channels
            :param out_channels: output convolution channels

        Testing: True
        """
        super(MultiBoxHeadModule, self).__init__()
        self._input_channels = in_channels
        self._output_channels = out_channels
        self._mid_channels = min((self._input_channels, middle_res_channel))
        # ------
        # convolution layer for `multi-box` header (use convolution 1x1)
        self.conv1 = local_convolution(in_channels=self._input_channels, out_channels=self._mid_channels)
        # ------
        # For like literal connection layers
        self.conv2 = local_convolution(in_channels=self._mid_channels, out_channels=self._mid_channels)
        self.conv3 = local_convolution(in_channels=self._mid_channels, out_channels=self._mid_channels)
        self.conv4 = local_convolution(in_channels=self._mid_channels, out_channels=self._output_channels, kernel_size=1, padding=0)
        

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)

        return x


def multi_box_cluster(out_channels, c, num_classes=2, input_channels=512):
    """
    Args:
        :param out_channels: output convolution channels
        :param c: config files TODO add configuration yaml for convolution neural network [level 1]
        :param num_classes: fully connected layer output hidden [fore-ground, back-ground]

    Testing: True
    """
    localization_layers, score_conf_layers = [], []
    for k_cls, channel in enumerate(out_channels):
        if k_cls == 0:
            localization_output_shape = 4
            score_conf_output_shape = 2
        elif k_cls == 1:
            localization_output_shape = 8
            score_conf_output_shape = 4
        else:
            localization_output_shape = 12
            score_conf_output_shape = 6
        localization_layers += [MultiBoxHeadModule(input_channels, c[k_cls] * localization_output_shape)]
        score_conf_layers += [MultiBoxHeadModule(input_channels, c[k_cls] * (num_classes + score_conf_output_shape))]

    return (localization_layers, score_conf_layers)
