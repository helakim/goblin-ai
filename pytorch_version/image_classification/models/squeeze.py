# *********************************************************************
# @Project    image_classification
# @FILE       squeeze.py
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
import torch
import torch.nn as nn
from ops.activation import SigmoidWeightedLinearUnits


def local_convolution(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=True)


class SeExcitation(nn.Module):

    def __init__(self, in_place, squeeze_place):
        """
        Args:
            :param in_place:
            :param squeeze_place:
        """
        super(SeExcitation, self).__init__()
        # ------
        # convolution unit reduce or expand (use 1x1 local convolution)
        self.in_place = in_place
        self.squeeze_place = squeeze_place
        self.reduce_expand = nn.Sequential(
            local_convolution(in_channel=self.in_place, out_channel=self.squeeze_place),  # reduce
            SigmoidWeightedLinearUnits(),
            local_convolution(in_channel=self.squeeze_place, out_channel=self.in_place),  # expand
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            :param x: input 2-d tensor
        """
        x_squeeze_units = self.reduce_expand(torch.mean(x, dim=(-2, -1), keepdims=True))

        return x_squeeze_units * x