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
import torch
import torch.nn as nn


class FeatureExtraction(nn.Module):

    def __init__(self, sub_module, extracted_layers):
        """
        Args:
            :param sub_module: named sub module (e.g. max pooling, convolution, dropout layer)
            :param extracted_layers: extraction weights by sub module
        """
        super(FeatureExtraction, self).__init__()
        # ------
        # TODO: global up sampling row, middle, high-resolution sub-pixels [level: 3]
        self.submodule = sub_module
        self.extracted_layers = extracted_layers

    def forward(self, x: torch.Tensor):
        outputs_feature = []
        # ------
        # named modules
        for name, module in self.submodule._modules.items():
            if name == 'classifier':
                # fully connected layers
                x = x.view(x.size(0), -1)

            if name == 'base':
                for block_name, cnn_block in module._modules.items():
                    x = cnn_block(x)
                    # -------
                    # extraction feature maps (fc7)
                    if block_name in self.extracted_layers:
                        outputs_feature.append(x)

        return outputs_feature