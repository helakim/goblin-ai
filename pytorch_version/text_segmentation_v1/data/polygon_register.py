# *********************************************************************
# @Project    goblin-ai
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
import os
import glob
import pathlib
import pyclipper
import cv2
import numpy as np
from torch.utils import data
from ops.augment import DataAugmentor


def polygons_area_validator(polygons, input_shape):
    """
    Args:
        :param polygons: input polygons area [x_1, y_1, ..., x_4, y_4]
        :param input_shape: image width height
    """
    # ------
    # check so that the text polygons area is the same direction and also filter some invalid polygons
    (height, width) = input_shape
    if polygons.shape[0] == 0:
        return polygons
    validated_polygons = []
    # x coordinate not max width `-1` and not min `zero`
    polygons[:, :, 0] = np.clip(polygons[:, :, 0], 0, width - 1)
    # y coordinate not max height `-1` and not min `zero`
    polygons[:, :, 1] = np.clip(polygons[:, :, 1], 0, height - 1)
    for poly in polygons:
        poly_contour = cv2.contourArea(poly)
        if abs(poly_contour) < 1:
            continue
        else:
            validated_polygons.append(poly)

    return np.array(validated_polygons)


def mask_map_generator(image_size, text_polygons, text_tags, training_mask, i, n, m):
    """
    Args
        :param image_size: image of height and width
        :param text_polygons: a finite number of straight line segments connected to from a closed polygonal chain (or polygonal circuit)
        :param text_tags: mask whether the text box is involved in training
        :param training_mask: training polygons areas
    """
    height, width = image_size
    score_map = np.zeros((height, width), dtype=np.uint8)
    for poly, tag in zip(text_polygons, text_tags):
        poly = poly.astype(np.int)
        r_i = 1 - (1 - m) * (n - i) / (n - 1)
        d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, closed=True)
        # ------
        # pco
        # add a path to clipper offset object in preparation for offsetting
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        # ------
        # this method takes two parameters, first is the structure that will receive the result of the offset operation
        # the second parameter is the amount to which the supplied paths will be offset -negative delta values to shrink
        # and polygons and polygons and positive delta to expand them
        """ CC
        #include "clipper.hpp"  
        using namespace ClipperLib;

        int main()
        {
          Path subj;
          Paths solution;
          subj << 
                IntPoint(348,257) << IntPoint(364,148) << IntPoint(362,148) << 
                IntPoint(326,241) << IntPoint(295,219) << IntPoint(258,88) << 
                IntPoint(440,129) << IntPoint(370,196) << IntPoint(372,275);
          ClipperOffset co;
          co.AddPath(subj, jtRound, etClosedPolygon);
          co.Execute(solution, -7.0);

          //draw solution ...
          DrawPolygons(solution, 0x4000FF00, 0xFF009900);
        }
        """
        s_poly = np.array([pco.Execute(-d_i)])
        cv2.fillPoly(score_map, s_poly, color=1)
        if tag:
            cv2.fillPoly(training_mask, s_poly, color=0)

    return score_map, training_mask


def augmentation(img, text_polygons, scales):
    augment_ops = DataAugmentor()
    img, text_polygons = augment_ops.random_scale(img, text_polygons, scales)

    return img, text_polygons


def image_label_generator(img_fn, text_polygons, text_tags, n, m, input_size, degrees=15, scales=None):
    """
    Args:
        :param img_fn: input origin image
        :param text_polygons: input polygons area [x_1, y_1, ..., x_4, y_4]
        :param text_tags: y labels
    """
    # ------
    # get image corresponding matrix and ground truth
    augment_ops = DataAugmentor()
    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    # ------
    # check out of bounding box (text instance)
    text_polygons = polygons_area_validator(text_polygons, (height, width))
    img, text_polygons = augmentation(img, text_polygons, scales, degrees, input_size)
    print(text_polygons)
    # ------
    # guaranteed short side >= input_size
    height, width, _ = img.shape
    min_short_edge = min(height, width)  # default (640, 640)
    if min_short_edge < input_size:
        scale = input_size / min_short_edge
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        text_polygons *= scale

    # ------
    # the feature map `F` is projected into `n` branches for all the text instances at a certain scale
    # among these masks, `S1` gives the segmentation result for the text instances with smallest scales
    # and S(n) denotes for the original segmentation mask expansion algorithm to gradually expand all the instances
    # kernels in `S1` to their complete shapes in S(n) and obtain the find detection result as `R`
    height, width, _ = img.shape
    training_mask = np.ones((height, width), dtype=np.uint8)
    score_maps = []
    for i in range(1, n + 1):
        # s1 -> s(n) from small to large
        score_map, training_mask = mask_map_generator(image_size=(height, width), text_polygons=text_polygons,
                                                      text_tags=text_tags, training_mask=training_mask, i=i, n=n, m=m)
        score_maps.append(score_map)
    score_maps = np.array(score_maps, dtype=np.float32)
    imgs = augment_ops.random_crop_origin([img, score_maps.transpose((1, 2, 0)), training_mask],
                                          (input_size, input_size))

    return imgs[0], imgs[1].transpose((2, 0, 1)), imgs[2]


class TextDataRegister(data.Dataset):

    def __init__(self, data_dir, data_shape=1240, n=2, m=0.7862, transform=None, target_transform=None):
        """
        Args:
            :param data_dir: root data dictionary [e.g. samples/]
            :param data_shape: input shape (e.g. 1240, 800)
            :param m: the minimal scale ratio which is a value (0, 1)
            :param transform: image transform (rotation, crop, brightness, etc...)
        """
        self.data_list = self.load_data(data_dir)
        self.data_shape = data_shape
        self.transform = transform
        self.target_transform = target_transform
        self.n = n
        self.m = m

    def __getitem__(self, item):
        img_path, text_polygons, text_tags = self.data_list[item]
        img, score_maps, training_mask = image_label_generator(img_path, text_polygons, text_tags, self.n, self.m,
                                                               self.data_shape)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            score_maps = self.target_transform(score_maps)
            training_mask = self.target_transform(training_mask)

        return img, score_maps, training_mask

    def __len__(self):
        return len(self.data_list)

    def load_data(self, data_dir):
        train_validation_data_list = []
        for idx in glob.glob(data_dir + '/imgs/*.jpg', recursive=True):
            per_sample = pathlib.Path(idx)
            label_path = os.path.join(data_dir, 'ground_truth', (str(per_sample.stem) + '.txt'))
            points, text = self._get_annotation(label_path)
            if len(points) > 0:
                train_validation_data_list.append((idx, points, text))
            else:
                print('[@] there is no suit points on {}'.format(label_path))

        return train_validation_data_list

    @staticmethod
    def _get_annotation(label_path):
        points = []
        text_tags = []
        with open(label_path, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                # ------
                # we training MLT-2015, 2017 dataset
                contents = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    label = contents[:-1]
                    if label == '*' or label == '###':
                        text_tags.append(True)
                    else:
                        text_tags.append(False)
                    x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = list(map(float, contents[:8]))
                    points.append([[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]])
                except RuntimeError:
                    print('[@] load label failed on {}'.format(label_path))

        return np.array(points, dtype=np.float32), np.array(text_tags, dtype=np.bool)
