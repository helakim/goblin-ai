# *********************************************************************
# @Project    image_classification
# @FILE       activation.py
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

import matplotlib.pylab as plt


class SigmoidWeightedLinearUnits(nn.Module):

    def forward(self, input_x):
        """
        Args:
            :param input_x: where * means, any number of additional dimensions (Number of Batch, vector)
            :return: same shape as the input_x (number of vector, vector)

        Description:
            - Applies the sigmoid linear unit function element-wise

        Feature:
            1. differentiable and do not need the manual implementation of the backward step (gradient descent)
            2. do not have any model trainable parameters, their parameters should be set in advance :)
        """

        return input_x * torch.sigmoid(input_x)


if __name__ == '__main__':
    one_dim_tensor = torch.linspace(-25, 25)
    sigmoid_weighted_linear_units = SigmoidWeightedLinearUnits()
    swiu_graph = sigmoid_weighted_linear_units(one_dim_tensor)
    relu_graph = torch.relu(one_dim_tensor)
    tanh = torch.tanh(one_dim_tensor)
    
    plt.title('compare results (swiu vs relu)')
    plt.plot(one_dim_tensor.numpy(), swiu_graph.numpy(), label='sigmoid_weighted_linear_units')
    plt.plot(one_dim_tensor.numpy(), relu_graph.numpy(), label='relu')
    plt.plot(one_dim_tensor.numpy(), tanh.numpy(), label='tanh')
    plt.legend()
    plt.show()