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
import torch.nn.functional as F
from models.layer import Bottleneck, local_convolution


class Network(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(Network, self).__init__()
        self.channels = [64, 128, 256, 512]
        self.inplanes = 64
        # ------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        # ------
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ------
        # 1.build residual convolution_layer
        self.layer1 = self.build_layer(block, self.channels[0], blocks=layers[0])
        # ------
        # 2.build residual convolution_layer
        self.layer2 = self.build_layer(block, self.channels[1], blocks=layers[1], stride=2)
        # ------
        # 3.build residual convolution_layer
        self.layer3 = self.build_layer(block, self.channels[2], blocks=layers[2], stride=2)
        # ------
        # 4.build residual convolution_layer
        self.layer4 = self.build_layer(block, self.channels[3], blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def build_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        stacked_layer = []
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_sample = nn.Sequential(
                local_convolution(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(num_features=planes * block.expansion)
            )

        stacked_layer.append(block(self.inplanes, planes, stride, down_sample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            stacked_layer.append(block(self.inplanes, planes))

        return nn.Sequential(*stacked_layer)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def model_call(**kwargs):
    m = Network(Bottleneck, [3, 4, 6, 3], **kwargs)

    return m


class ResidualNetwork(nn.Module):

    def __init__(self, num_classes, **kwargs):
        """
        Args:
            :param num_classes: fully connected output hidden units
        """
        super(ResidualNetwork, self).__init__()
        cnn_backbone = model_call(**kwargs)
        # ------
        # feature extraction layer (reduce last 2 layers)
        self.base = nn.Sequential(*list(cnn_backbone.children()))[:-2]
        # ------
        # fully connected layer with soft max layer (classifier)
        self.feat_dim = 512 * 4

        # ------
        # feature extraction layers
        self.conv1 = nn.Conv2d(in_channels=self.feat_dim, out_channels=128, kernel_size=1, stride=1, padding=0,bias=True)
        self.bn = nn.BatchNorm2d(self.feat_dim)
        self.relu = nn.ReLU(inplace=True)
        # ------
        # fc7 layers (obtain a alpha weights)
        self.classifier = nn.Linear(in_features=self.feat_dim, out_features=num_classes)

    def forward(self, x):
        global last_feature_map
        x = self.base(x)
        last_feature_map = last_feature_map.view(last_feature_map.size()[0:3])
        last_feature_map = last_feature_map / torch.pow(last_feature_map, 2).sum(dim=1,keepdim=True).clamp(min=3e-11).sqrt()
        # ------
        # Computed Convolution Filter -> Function Space Transform -> Vector Space
        x = F.avg_pool2d(x, x.size()[2:])
        fn_space = x.view(x.size(0), -1)

        return fn_space, last_feature_map