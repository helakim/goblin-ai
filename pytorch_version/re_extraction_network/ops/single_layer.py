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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def pool_2d(tensor):
    """
    Args:
        :param tensor: input 2-dimensional tensor (extraction convolution weights)
        :param pool_type: pooling type [chooses max pooling, average pooling]
    """
    input_spec = tensor.size()
    dilated_kernel = (np.int(input_spec[2] / 8), np.int(input_spec[3]))
    x = F.max_pool2d(tensor, kernel_size=dilated_kernel)

    x = x[0].cpu().data.numpy()
    x = np.transpose(x, (2, 1, 0))[0]

    return x


class ChannelWiseMaxPool(nn.Module):
    def __init__(self, pool_type='dilated_kernel'):
        super(ChannelWiseMaxPool, self).__init__()
        self.pool_type = pool_type

    def forward(self, input_tensor):
        """
        Args:
            :param input_tensor: input 2-dimensional tensor (extraction convolution weights)
        """
        x_spec = input_tensor.size()

        if self.dilated_kernel == 'dilated_kernel':
            kernel_size = (1, x_spec[3])
            pooled = F.max_pool2d(input=x_spec, kernel_size=kernel_size)
        elif self.dilated_kernel == 'atrous_kernel':
            kernel_size = (3, x_spec[3])
            pooled = F.avg_pool2d(input=x_spec, kernel_size=kernel_size)

        return pooled
