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
import math
import torch
from torchvision.ops.boxes import nms
from cfg.cv_utils import prediction_coords, cuda_tensor_convert


def soft_nms(boxes, scores, overlap=0.5, top_k=200):
    """
    Args:
        boxes: (tensor) The location predictions for the img, Shape: [num_priors,4].
        scores: (tensor) The class prediction scores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box predictions to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)

    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()

    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h

        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    return keep, count


class DetectionHead(object):
    def __init__(self, variance):
        super(DetectionHead, self).__init__()
        self.variance = variance

    def forward(self, location_tensor, conf_score, prior_boxes, conf_thresh, nms_thresh):
        """
        Args:
            :param location_tensor:
            :param conf_score:
            :param prior_boxes:
            :param conf_thresh:
            :param nms_thresh:

        Testing: True
        """
        # ------
        # local variables
        num_classes = 2
        max_value = 1

        batch_size = location_tensor.size(0)
        num_prior_boxes = prior_boxes.size(0)
        score_predictions = conf_score.view(batch_size, num_prior_boxes, num_classes).transpose(2, 1)

        bbox_buckets = []

        for idx in range(batch_size):
            basic_box = prior_boxes  # default box
            # ------
            # decoding bounding boxes
            decode_coords = prediction_coords(location_tensor[idx], priors_bbox=basic_box, variances=self.variance)
            # -------------------------- #
            confidence_scores = score_predictions[idx, max_value]  # choose one
            indices = (confidence_scores >= conf_thresh).nonzero().squeeze()
            decode_coords = decode_coords[indices]

            confidence_scores = confidence_scores[indices]
            # -------------------------- #
            if confidence_scores.dim() == 0:
                # sorted by confidence scores [one of bounding boxes]
                # bbox_buckets.append(np.zeros(0, 5))
                bbox_buckets.append(torch.empty(0, 5))
                continue

            # TODO: add Description of Non Max Suppression [level: 1]
            # keep_index, _ = soft_nms(decode_coords, confidence_scores)
            keep_index = nms(boxes=decode_coords, scores=confidence_scores, iou_threshold=nms_thresh)
            scores = confidence_scores[keep_index].view(1, -1, 1)
            bounding_boxes = decode_coords[keep_index].view(1, - 1, 4)
            # ------
            # tensor shape [(N, score), (N, x_min, y_min, x_max, y_max)]
            sorted_bounding_boxes = torch.cat((scores, bounding_boxes), dim=-1)
            bbox_buckets.append(sorted_bounding_boxes)
        if batch_size == 1:
            return bbox_buckets[0]
        bbox_buckets = torch.cat(bbox_buckets, dim=0)

        return bbox_buckets


class ComputePriorCenterOffset(object):

    def __init__(self, image_size, feature_maps):
        """
        Args:
            :param image_size: input image size
            :param feature_maps: input feature map (required by convolution modules)

        Testing: True
        """
        super(ComputePriorCenterOffset, self).__init__()
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.num_priors = 6
        self.variance = [0.1, 0.2]
        self.min_sizes = [16, 32, 64, 128, 256, 512]
        self.max_sizes = []

        self.stride_size = [4, 8, 16, 32, 64, 128]
        self.aspect_ratios = [[1.5], [1.5], [1.5], [1.5], [1.5], [1.5]]
        self.clip = True  # make default bounding boxes in [0, 1]

    def forward(self):
        mean_value = []

        for identity, feature in enumerate(self.feature_maps):
            for i_ in range(feature[0]):
                for j_ in range(feature[1]):
                    feature_identity_i_ = self.image_size[0] / self.stride_size[identity]
                    feature_identity_j_ = self.image_size[1] / self.stride_size[identity]
                    # ------
                    # center x,y => aspect ratio 1
                    center_x = (j_ + 0.5) / feature_identity_j_
                    center_y = (i_ + 0.5) / feature_identity_i_

                    rel_identity_i_ = self.min_sizes[identity] / self.image_size[1]
                    rel_identity_j_ = self.min_sizes[identity] / self.image_size[0]

                    if len(self.aspect_ratios[0]) == 0:
                        mean_value += [center_x, center_y, rel_identity_i_, rel_identity_j_]

                    if len(self.max_sizes) == len(self.min_sizes):
                        rel_identity_prime_i = math.sqrt(rel_identity_i_ * (self.max_sizes[identity] / self.image_size[1]))
                        rel_identity_prime_j = math.sqrt(rel_identity_j_ * (self.max_sizes[identity] / self.image_size[0]))

                        mean_value += [center_x, center_y, rel_identity_prime_i, rel_identity_prime_j]
                    # ------
                    # initialization of aspect ratio
                    for aspect_ratio in self.aspect_ratios[identity]:
                        if len(self.max_sizes) == len(self.min_sizes):
                            mean_value += [center_x, center_y, rel_identity_prime_i / math.sqrt(aspect_ratio), rel_identity_prime_j * math.sqrt(aspect_ratio)]
                        # mean_value += [center_x, center_x, rel_identity_i_ / math.sqrt(aspect_ratio), rel_identity_j_ * math.sqrt(aspect_ratio)]
                        mean_value += [center_x, center_y, rel_identity_i_ / math.sqrt(aspect_ratio), rel_identity_j_ * math.sqrt(aspect_ratio)]

        res = torch.Tensor(mean_value).view(-1, 4)
        res = cuda_tensor_convert(res)
        # ------
        # deep learning model lovers normalizations
        res.clamp_(max=1, min=0)

        return res
