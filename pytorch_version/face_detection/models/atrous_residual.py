# *********************************************************************
# @Project    goblin-ai
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   29/09/2019
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
import functools
import numpy as np
import torch
import torch.nn as nn


class AtrousResidualNetwork(nn.Module):

    def __init__(self, origin_residual, atrous_aspect_ratio):
        """
        Args:
            :param origin_residual: original residual units
            :param atrous_aspect_ratio: convolution units for atrous aspect ratio
        """
        super(AtrousResidualNetwork, self).__init__()
        self.atrous_aspect_ratio = np.int(atrous_aspect_ratio)
        filter_size = [8, 16]
        if self.atrous_aspect_ratio == filter_size[0]:
            # ------
            # residual layers 3 and 4
            origin_residual.layers_3.apply(functools.partial(self._atrous_reduce, atrous=2))
            origin_residual.layers_4.apply(functools.partial(self._atrous_reduce, atrous=4))
        elif self.atrous_aspect_ratio == filter_size[1]:
            # ------
            # only residual layer 4
            origin_residual.layers_4.apply(functools.partial(self._atrous_reduce, atrous=2))
        else:
            pass
        # ------
        # extraction deep residual network (average pooling layer and fully convolution layer)

        # -------
        # first convolution module
        self.convolution_1_unit = origin_residual.convolution_1
        self.bn_1 = origin_residual.bn_1
        self.relu_1 = origin_residual.relu_1
        # -------
        # second convolution module
        self.convolution_2_unit = origin_residual.convoltuion_2
        self.bn_2 = origin_residual.bn_2
        self.relu_2 = origin_residual.relu_2
        # -------
        # last convolution module
        self.convolution_3_unit = origin_residual.convoltuion_3
        self.bn_3 = origin_residual.bn_3
        self.relu_3 = origin_residual.relu_3
        # ------
        # stacked residual layers [1, 2, 3, 4]
        self.max_pool = origin_residual.max_pool
        self.layer_1 = origin_residual.layer_1
        self.layer_2 = origin_residual.layer_2
        self.layer_3 = origin_residual.layer_3
        self.layer_4 = origin_residual.layer_4

    def _atrous_reduce(self, class_module, atrous):
        # ------
        # named class for PyTorch Model
        named_cls = class_module.__class__.__name__

        if named_cls.find('Conv') != -1:
            if class_module.stride(2, 2):
                class_module.stride(1, 1)
                if class_module.kernel_size == (3, 3):
                    class_module.atrous = (atrous // 2, atrous // 2)
                    class_module.padding = (atrous // 2, atrous // 2)
            else:
                if class_module.kernel_size == (3, 3):
                    class_module.atrous = (atrous, atrous)
                    class_module.padding = (atrous, atrous)

    def forward(self, input_x, recycle_features_maps=False):
        # ------
        # feature extraction layer
        x_1 = self.relu_1(self.bn_1(self.convolution_1_unit(input_x)))
        x_2 = self.relu_2(self.bn_2(self.convolution_2_unit(x_1)))
        x_3 = self.relu_3(self.bn_3(self.convolution_3_unit(x_2)))
        # ------
        # activation maps layers
        x = self.max_pool(x_3)
        layer_1 = self.layer_1(x)
        layer_2 = self.layer_2(layer_1)
        layer_3 = self.layer_3(layer_2)
        layer_4 = self.layer_4(layer_3)
        # ------
        # concatenate layers
        layer_concat = torch.cat((layer_1, layer_2, layer_3, layer_4), dim=0)
        if recycle_features_maps:
            return layer_concat

        return layer_4
