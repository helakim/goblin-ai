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
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from ops.bbox import DetectionHead, ComputePriorCenterOffset
from models.convolutions import FeatureEnhanceModule
from models.header import multi_box_cluster


def load_backbone(name='default'):
    if name == 'default':
        # TODO: Add backbone model [level: 1]
        return torchvision.models.resnet152(pretrained=False)
    else:
        pass


def local_convolution_1x1(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    unit_x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    return unit_x


class SingleShotDetector(nn.Module):

    def __init__(self, c=None):
        """
        Description:
            - Single Short Multiple box is composed of a base Very Deep Convolution Neural Network followed by the added `multi-box`
            convolution layers, each multiple convolution layer branches into 2-dimensional convolution unit for classes confidence
            scores (background or foreground) and 2-dimensional convolution unit for localization(locations) predictions (for like
            feature pyramid network) and associated prior-boxes layer(back-prop) to produce basic bounding boxes, anchors specific
            to the layers feature map size

        Args:
            :param c: hyper parameters for convolution neural network
        """
        # -------
        # output_channels for high resolution]
        super(SingleShotDetector, self).__init__()
        channels = [256, 512, 1024, 2048, 512, 256]
        feature_pyramid_channels = context_pyramid_channels = channels
        self.prior_cache = {}
        self.num_fully_connected_layers = 2  # foreground and background
        backbone = load_backbone(name='default')

        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        # ------
        # input_shape = [56 X 56]
        # | convolution: 1 X 1, filter: 64  |
        # | convolution: 3 X 3, filter: 64  | X 3 blocks
        # | convolution: 1 X 1, filter: 256 |
        self.layer2 = nn.Sequential(backbone.layer2)
        # input_shape = [28 X 28]
        # | convolution: 1 X 1, filter: 128 |
        # | convolution: 3 X 3, filter: 128 | X 8 blocks
        # | convolution: 1 X 1, filter: 512 |
        self.layer3 = nn.Sequential(backbone.layer3)
        # input_shape = [14 X 14]
        # | convolution: 1 X 1, filter: 256  |
        # | convolution: 3 X 3, filter: 256  | X 36 blocks
        # | convolution: 1 X 1, filter: 1024 |
        self.layer4 = nn.Sequential(backbone.layer4)
        # input_shape = [7 X 7]
        # | convolution: 1 X 1, filter: 512  |
        # | convolution: 3 X 3, filter: 512  |
        # | convolution: 1 X 1, filter: 2046 |
        self.layer5 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(512), nn.ReLU(inplace=True)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        # Feature Pyramid Network
        self.latlayer3 = local_convolution_1x1(in_channels=feature_pyramid_channels[3], out_channels=feature_pyramid_channels[2])
        self.latlayer2 = local_convolution_1x1(in_channels=feature_pyramid_channels[2], out_channels=feature_pyramid_channels[1])
        self.latlayer1 = local_convolution_1x1(in_channels=feature_pyramid_channels[1], out_channels=feature_pyramid_channels[0])
        # ------
        # Smooth layer (middle resolution)
        self.smooth3 = local_convolution_1x1(in_channels=feature_pyramid_channels[2], out_channels=feature_pyramid_channels[2])
        self.smooth2 = local_convolution_1x1(in_channels=feature_pyramid_channels[1], out_channels=feature_pyramid_channels[1])
        self.smooth1 = local_convolution_1x1(in_channels=feature_pyramid_channels[0], out_channels=feature_pyramid_channels[0])
        # ------
        # A novel Feature Enhance Module to utilize different level information and thus obtain more discriminable and robust features
        self.cpm3_3 = FeatureEnhanceModule(context_pyramid_channels[0])
        self.cpm4_3 = FeatureEnhanceModule(context_pyramid_channels[1])
        self.cpm5_3 = FeatureEnhanceModule(context_pyramid_channels[2])
        self.cpm7 = FeatureEnhanceModule(context_pyramid_channels[3])
        self.cpm6_2 = FeatureEnhanceModule(context_pyramid_channels[4])
        self.cpm7_2 = FeatureEnhanceModule(context_pyramid_channels[5])
        # ------
        # obtain a localization layers and confidence score layers
        clustered_multi_box_head = multi_box_cluster(channels, [1, 1, 1, 1, 1, 1, 1], self.num_fully_connected_layers)
        self.loc = nn.ModuleList(clustered_multi_box_head[0])
        self.conf = nn.ModuleList(clustered_multi_box_head[1])
        self.softmax = nn.Softmax(dim=-1)
        self.detect = DetectionHead([0.1, 0.2])

    def init_priors(self, feature_maps, image_size):
        """
            :param feature_maps: 2-dimensional feature maps
            :param image_size: input image size
        """
        module_keys = ".".join([str(item) for i in range(len(feature_maps)) for item in feature_maps[i]]) + "," + ".".join([str(_) for _ in image_size])
        if module_keys in self.prior_cache:
            return self.prior_cache[module_keys].clone()

        cal_center_offset = ComputePriorCenterOffset(image_size=image_size, feature_maps=feature_maps)
        bbox_prior = cal_center_offset.forward()

        self.prior_cache[module_keys] = bbox_prior.clone()

        return bbox_prior

    def forward(self, x, confidence_threshold, nms_threshold=0.5):
        image_size = [x.shape[2], x.shape[3]]  # width, height
        localization, score_conf = list(), list()
        activation_maps = []
        module_bbox_num = 1
        # ------
        # Extraction deep residual network (for like U-net)
        convolution_3 = self.layer1(x)
        convolution_4 = self.layer2(convolution_3)
        convolution_5 = self.layer3(convolution_4)

        fully_convolution = self.layer4(convolution_5)

        convolution_6 = self.layer5(fully_convolution)
        convolution_7 = self.layer6(convolution_6)
        # ------------------------------------------------------------ #
        # Feature Pyramid Network
        # literal_connection_fpn_3 = F.interpolate(self.latlay3(fully_convolution), size=self.smooth3(convolution_5).shape[2:], mode='bilinear', align_corners=True)
        # literal_connection_fpn_3 = literal_connection_fpn_3 * self.smooth3(convolution_5)
        #
        # literal_connection_fpn_2 = F.interpolate(self.latlay2(literal_connection_fpn_3), size=self.smooth2(convolution_4).shape[2:], mode='bilinear', align_corners=True)
        # literal_connection_fpn_2 = literal_connection_fpn_2 * self.smooth2(convolution_4)
        #
        # literal_connection_fpn_1 = F.interpolate(self.latlay1(literal_connection_fpn_2), size=self.smooth1(convolution_3).shape[2:], mode='bilinear', align_corners=True)
        # literal_connection_fpn_1 = literal_connection_fpn_1 * self.smooth1(convolution_3)
        # ------------------------------------------------------------ #
        literal_connection_fpn_3 = self.__like_up_sampling(self.latlayer3(fully_convolution), self.smooth3(convolution_5))
        literal_connection_fpn_2 = self.__like_up_sampling(self.latlayer2(literal_connection_fpn_3), self.smooth2(convolution_4))
        literal_connection_fpn_1 = self.__like_up_sampling(self.latlayer1(literal_connection_fpn_2), self.smooth1(convolution_3))
        # ------
        # Pyramid level feature maps (P3, P4, P5)
        convolution_5 = literal_connection_fpn_3
        convolution_4 = literal_connection_fpn_2
        convolution_3 = literal_connection_fpn_1

        # ------
        # Concatenated Convolution Module
        # concat_conv_modules = torch.cat((feature_extraction_conv_3, feature_extraction_conv_4, feature_extraction_conv_5, fully_convolution, convolution_6, convolution_7))
        concat_feature_maps = [convolution_3, convolution_4, convolution_5, fully_convolution, convolution_6, convolution_7]
        concat_feature_maps[0] = self.cpm3_3(concat_feature_maps[0])
        concat_feature_maps[1] = self.cpm4_3(concat_feature_maps[1])
        concat_feature_maps[2] = self.cpm5_3(concat_feature_maps[2])
        # ------
        # Fully_convolution layers
        concat_feature_maps[3] = self.cpm7(concat_feature_maps[3])
        concat_feature_maps[4] = self.cpm6_2(concat_feature_maps[4])
        concat_feature_maps[5] = self.cpm7_2(concat_feature_maps[5])
        # ------
        # Cal single shot detector `multi-box`
        for (tensor_, loc_, c) in zip(concat_feature_maps, self.loc, self.conf):
            activation_maps.append([tensor_.shape[2], tensor_.shape[3]])
            localization.append(loc_(tensor_).permute(0, 2, 3, 1).contiguous())

            len_conf = len(score_conf)
            cls = self.max_in_out(c(tensor_), len_conf)

            score_conf.append(cls.permute(0, 2, 3, 1).contiguous())
        # -------
        # Matcher expansion pyramid anchor
        object_localization = torch.cat([x[:, :, :, :4 * module_bbox_num].contiguous().view(x.size(0), -1) for x in localization], dim=1)
        object_confidence_scores = torch.cat([x[:, :, :, :2 * module_bbox_num].contiguous().view(x.size(0), -1) for x in score_conf], dim=1)

        self.priors = self.init_priors(activation_maps, image_size)
        conf_predictions = self.softmax(object_confidence_scores.view(object_confidence_scores.size(0), -1, self.num_fully_connected_layers))
        object_localization = object_localization.view(object_localization.size(0), -1, 4)

        output = self.detect.forward(object_localization, conf_predictions, self.priors, confidence_threshold, nms_threshold)

        return output

    @staticmethod
    def __like_up_sampling(x, y):
        """
        Args:
            :param x: feature map to be `up sample` (bottom-up)
            :param y: lateral connection feature map (top-down)
        """
        maps = F.interpolate(input=x, size=y.shape[2:], mode='bilinear', align_corners=True) * y

        return maps


    def max_in_out(self, box, length_conf):
        """
        Args:
            :param box: input bounding boxes
            :param length_conf: score buckets

        Testing: True
        """
        c = torch.chunk(box, box.shape[1], dim=1)
        bbox_max = torch.max(torch.max(c[0], c[1]), c[2])
        cls = (torch.cat([bbox_max, c[3]], dim=1))if length_conf == 0 else torch.cat([c[3],bbox_max], dim=1)

        if len(c) == 6:
            cls = torch.cat((cls, c[4], c[5]), dim=1)
        elif len(c) == 8:
            cls = torch.cat((cls, c[4], c[5], c[6], c[7]), dim=1)

        return cls




