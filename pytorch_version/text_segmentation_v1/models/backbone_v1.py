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
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import tqdm


def convolution3x3(in_channel, out_channel, stride=1):
    """ 3x3 convolution layer
    """
    layer = nn.Conv2d(in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
    return layer


def convolution1x1(in_channels, out_channels, stride=1):
    """ 1x1 convolution layer
    """
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    return layer


class BasicBLock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBLock, self).__init__()
        # ------
        # convolution_layer_1
        self.conv_1 = convolution3x3(in_planes, planes, stride)
        self.bn_1 = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        # ------
        # convolution_layer_2
        self.conv_2 = convolution3x3(planes, planes)
        self.bn_2 = nn.BatchNorm2d(num_features=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual_unit = x
        # ------
        # convolution_module_1
        out = self.relu(self.bn_1(self.conv_1(x)))
        # ------
        # convolution_module_2
        out = self.bn_2(self.conv_2(out))
        # ------
        # down sample
        if self.downsample is not None:
            residual_unit = self.downsample(x)
        # ------
        # skip connection
        out += residual_unit
        out = self.relu(out)

        return out


class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleNeckBlock, self).__init__()
        # ------
        self.conv_1 = convolution1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        # ------
        self.conv_2 = convolution3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        # ------

        self.conv_3 = convolution1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(num_features=planes * self.expansion)
        # ------
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual_unit = x
        # ------
        # convolution_module_1
        out = self.relu(self.bn1(self.conv_1(x)))
        # ------
        # convolution_module_2
        out = self.relu(self.bn2(self.conv_2(out)))
        # ------
        # convolution_module_3
        out = self.bn3(self.conv_3(out))

        if self.downsample is not None:
            residual_unit = self.downsample(x)
        out += residual_unit
        out = self.relu(out)

        return out


class ResidualNetwork(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResidualNetwork, self).__init__()
        self.in_planes = 64
        c = [64, 128, 256, 512]
        # ------
        # convolution first layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ------
        # convolution with residual layer (1, 2, 3, 4)
        self.layer1 = self._build_layer(block, c[0], layers[0])
        self.layer2 = self._build_layer(block, c[1], layers[1], stride=2)
        self.layer3 = self._build_layer(block, c[2], layers[2], stride=2)
        self.layer4 = self._build_layer(block, c[3], layers[3], stride=2)

        # v1
        # ------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Suggest FaceBook ai-research
        # ------
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeckBlock):
                    nn.init.constant_(m.bn_3.weight, 0)
                elif isinstance(m, BasicBLock):
                    nn.init.constant_(m.bn_2.weight, 0)

    def _build_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                # nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                convolution1x1(self.in_planes, planes * block.expansion, stride=stride),
                nn.BatchNorm2d(num_features=planes * block.expansion),
            )

        stacked_layer = []
        stacked_layer.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for idx in range(1, blocks):
            stacked_layer.append(block(self.in_planes, planes))

        return nn.Sequential(*stacked_layer)

    def forward(self, input_x):
        # ------
        # first convolution stage
        x = self.relu(self.bn1(self.conv1(input_x)))
        # ------
        # required by feature pyramid layers
        # convolution module bottom-up, tom-down, skip-connection
        C1 = self.maxpool(x)
        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)

        return C1, C2, C3, C4, C5


def custom_deep_residual_50( **kwargs):
    """ TODO.md: Add description
    """
    model = ResidualNetwork(BottleNeckBlock, [3, 4, 6, 3], **kwargs)
    return model
