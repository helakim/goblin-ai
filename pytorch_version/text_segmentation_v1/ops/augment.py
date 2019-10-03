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
import torch
import math
import random
import cv2
import numpy as np
import skimage
import re


class E(object):

    def __init__(self, n, h):
        super(E, self).__init__()
        self.n = n
        self.h = h

    def __call__(self, x, y):
        height, width = x.size()[:2]
        __e = np.ones((height, width), np.float16)

        for idx in range(self.n):
            x = np.random.randint(width)
            y = np.random.randint(height)
            # -------
            # cliped x, y coordinates
            x_min = np.clip(x - self.n // 2.0, 0, width)
            y_min = np.clip(y - self.n // 2.0, 0, height)
            x_max = np.clip(x + self.n // 2.0, 0, width)
            y_max = np.clip(y + self.n // 2.0, 0, height)

            __e[y_min: y_max, x_min: x_max] = 0.0
        __e = torch.from_numpy(__e)
        __e = __e.expand_as(x)

        return x * __e


class DataAugmentor(object):
    """ TODO.md: Add description
    """

    def __init__(self, *args, **kwargs):
        super(DataAugmentor, self).__init__(*args)

    def verify_coordinates(coordi, r=1.5707963267948966):
        r_ = np.deg2rad(r)
        w_angle = np.abs(np.degrees(math.atan2(coordi[2] - coordi[0], coordi[3] - coordi[1])) - r_)
        if w_angle < 45:
            return [coordi[2], coordi[3], coordi[4], coordi[5], coordi[6], coordi[7], coordi[0], coordi[1]]
        else:
            return [coordi[0], coordi[1], coordi[2], coordi[3], coordi[4], coordi[5], coordi[6], coordi[7]]

    def perspective_transform_crop(self, ori_img, coordi):
        coordi = self.verify_coordinates(coordi)

        h, w = np.shape(ori_img)[:2]

        coordi[0] = np.clip(coordi[0], 0, w - 1)
        coordi[1] = np.clip(coordi[1], 0, h - 1)
        coordi[2] = np.clip(coordi[2], 0, w - 1)
        coordi[3] = np.clip(coordi[3], 0, h - 1)
        coordi[4] = np.clip(coordi[4], 0, w - 1)
        coordi[5] = np.clip(coordi[5], 0, h - 1)
        coordi[6] = np.clip(coordi[6], 0, w - 1)
        coordi[7] = np.clip(coordi[7], 0, h - 1)

        min_x = min(coordi[0], coordi[6])
        min_y = min(coordi[1], coordi[3])
        max_x = max(coordi[2], coordi[4])
        max_y = max(coordi[5], coordi[7])

        try:
            crop = ori_img[min_y:max_y, min_x:max_x]
            new_coordi = [coordi[0] - min_x, coordi[1] - min_y, coordi[2] - min_x, coordi[3] - min_y, coordi[4] - min_x,
                          coordi[5] - min_y, coordi[6] - min_x, coordi[7] - min_y]
            src_pts = np.float32(
                [[new_coordi[0], new_coordi[1]], [new_coordi[2], new_coordi[3]], [new_coordi[4], new_coordi[5]],
                 [new_coordi[6], new_coordi[7]]])
            dst_pts = np.float32([[0, 0], [100, 0], [100, 32], [0, 32]])

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            dst = cv2.warpPerspective(crop, M, (100, 32))
        except:
            dst = None

        return dst

    def special_symbols_check(self, string, coordinates):
        coordi = self.verify_coordinates(coordinates)

        crop_lt_x = min(coordi[0][0], coordi[3][0])
        crop_lt_y = min(coordi[0][1], coordi[1][1])
        crop_rt_x = max(coordi[1][0], coordi[2][0])
        crop_rt_y = max(coordi[2][1], coordi[3][1])
        crop_w = crop_rt_x - crop_lt_x
        crop_h = crop_rt_y - crop_lt_y

        # ------
        # coordinate clip
        offset_x = crop_lt_x
        offset_y = crop_lt_y

        coordi[0][0] -= offset_x
        coordi[0][1] -= offset_y
        coordi[1][0] -= offset_x
        coordi[1][1] -= offset_y
        coordi[2][0] -= offset_x
        coordi[2][1] -= offset_y
        coordi[3][0] -= offset_x
        coordi[3][1] -= offset_y

        blocks = []
        blocks_text = []
        block_flag = False
        before_ss_margin = 0
        weighted_size = 0
        block_s = 0
        block_e = 0
        b_text = ''
        ss_flag = False
        string_length = len(string)
        for i in range(string_length):
            if bool(re.search('[가-힣0-9a-zA-Z]', string[i])):
                b_text += string[i]
                if not block_flag:
                    block_s = weighted_size - before_ss_margin
                weighted_size += 4
                before_ss_margin = 0
                block_e = weighted_size
                block_flag = True
                if i == string_length - 1:
                    blocks_text.append(b_text)
                    blocks.append([block_s, block_e])
            elif bool(re.search('[\-_~@#$%&*+=?<>]', string[i])):
                if block_flag:
                    blocks_text.append(b_text)
                    b_text = ''
                    block_e += 0
                    blocks.append([block_s, block_e])
                weighted_size += 4
                before_ss_margin = 0
                block_flag = False
                ss_flag = True
            else:
                if block_flag:
                    blocks_text.append(b_text)
                    b_text = ''
                    block_e += 0
                    blocks.append([block_s, block_e])
                weighted_size += 2
                before_ss_margin = 0
                block_flag = False
                ss_flag = True

        if ss_flag:
            # perspective transform
            polygon_width = np.sqrt(((coordi[0][0] - coordi[1][0]) ** 2) + ((coordi[0][1] - coordi[1][1]) ** 2))
            polygon_height = np.sqrt(((coordi[0][0] - coordi[3][0]) ** 2) + ((coordi[0][1] - coordi[3][1]) ** 2))

            polygon_aspect_ratio = polygon_height / polygon_width

            trans_img_width = 300
            trans_img_height = int(trans_img_width * polygon_aspect_ratio)
            src_ref_pts = np.float32(coordi)
            dst_ref_pts = np.float32([[0, 0], [trans_img_width - 1, 0], [trans_img_width - 1, trans_img_height - 1],
                                      [0, trans_img_height - 1]])

            trans_matrix = cv2.getPerspectiveTransform(dst_ref_pts, src_ref_pts)
            blocks_pts = []
            for i in range(len(blocks)):
                blocks_pts.append([blocks[i][0] / weighted_size * trans_img_width, 0])
                blocks_pts.append([blocks[i][1] / weighted_size * trans_img_width, 0])
                blocks_pts.append([blocks[i][1] / weighted_size * trans_img_width, trans_img_height - 1])
                blocks_pts.append([blocks[i][0] / weighted_size * trans_img_width, trans_img_height - 1])
            blocks_pts = np.array([blocks_pts], dtype=np.float32)
            if np.shape(blocks_pts)[1]:
                t_blocks = np.around(cv2.perspectiveTransform(blocks_pts, trans_matrix)[0]).astype(int)
                t_blocks = np.stack([np.clip(t_blocks[:, 0], 0, crop_w - 1), np.clip(t_blocks[:, 1], 0, crop_h - 1)],
                                    axis=1)
                result_blocks = np.reshape(t_blocks, (-1, 4, 2))
            else:
                result_blocks = []
        else:
            result_blocks = np.reshape(coordi, (-1, 4, 2))

        result_rects = []
        for result_block in result_blocks:
            result_block[0][0] += offset_x
            result_block[0][1] += offset_y
            result_block[1][0] += offset_x
            result_block[1][1] += offset_y
            result_block[2][0] += offset_x
            result_block[2][1] += offset_y
            result_block[3][0] += offset_x
            result_block[3][1] += offset_y

            # ------
            # original coordinate
            result_lt_x = min(result_block[0][0], result_block[3][0])
            result_lt_y = min(result_block[0][1], result_block[1][1])
            result_rt_x = max(result_block[1][0], result_block[2][0])
            result_rt_y = max(result_block[2][1], result_block[3][1])
            # ------
            # we got a minima coordinate
            # result_lt_x = (result_block[0][0] + result_block[3][0]) * 0.05
            # result_lt_y = min(result_block[0][1], result_block[1][1])
            # result_rt_x = (result_block[1][0] + result_block[2][0]) * 0.05
            # result_rt_y = max(result_block[2][1], result_block[3][1])

            result_rects.append([result_lt_x, result_lt_y, result_rt_x, result_rt_y])

        return zip(blocks_text, result_rects)

    def noise_box_search(self, bboxes_inform, minimum_side_length=15):
        bboxes_inform_result = []
        for bbox_inform in bboxes_inform:
            results = self.special_symbols_check(bbox_inform[8], bbox_inform[:8])
            for result in results:
                # Add 't', 'T',

                if len(result[0]) == 0 and result[0] != 't':
                    continue

                coordi = np.reshape(result[1], (-1))
                bbox_w = coordi[2] - coordi[0]
                bbox_h = coordi[3] - coordi[1]

                margin = bbox_h * 0.04
                coordi[1] += margin
                coordi[3] -= margin

                margin = bbox_w * 0.04
                coordi[0] += margin
                coordi[2] -= margin

                # too small
                if bbox_w < minimum_side_length or bbox_h < minimum_side_length:
                    continue

                bboxes_inform_result.append([coordi[0], coordi[1], coordi[2], coordi[3], result[0].upper()])

        return bboxes_inform_result

    @staticmethod
    def add_gaussian_noise(img):
        noise = skimage.util.random_noise(image=img, mode='gaussian', clip=True)  # gaussian `2-D`
        random_gaussian_noise = (noise * 255).astype(img.dtype)

        return random_gaussian_noise

    @staticmethod
    def random_scale(img, text_polygons, scales):
        tmp_text_polygons = text_polygons.copy()
        selected_scale = float(np.random.choice(scales))
        # ------
        # resize image use selected scale (x, y)
        img = cv2.resize(img, dsize=None, fx=selected_scale, fy=selected_scale)
        tmp_text_polygons *= selected_scale

        return img, tmp_text_polygons

    @staticmethod
    def random_rotate_img_bbox(img, text_polygons, degrees, same_size=False):
        """
        Args:
            :param img: read image matrix (height, width, channels)
            :param text_polygons: a finite number of straight line segments connected to from a closed polygonal chain (or polygonal circuit)
            :param degrees: angle can be a numeric as big as the original image
            :param same_size: width, height
        """
        # ------
        height = img.shape[0]
        width = img.shape[1]
        angle = np.random.uniform(degrees[0], degrees[1])

        if same_size:
            new_width = width
            new_height = height
        else:
            # ------
            # step1. angle arcing, convert angles from radians to degrees
            # step2. calculate the image of height, width after rotation
            radians = np.deg2rad(angle)
            new_height = (abs(np.cos(radians) * height) + abs(np.sin(radians) * width))
            new_width = (abs(np.sign(radians) * height) + abs(np.cos(radians) * width))
        # ------
        # constructing an affine matrix
        # TODO.md: check matrix center to tuple [level: 3]
        rotate_matrix = cv2.getRotationMatrix2D((new_width * 0.5, new_height * 0.5), angle, 1)
        # ------
        # calculate the offset from the center point of the original
        # image to the center ponits of the `target_image` (rotate affine matirx)
        rotate_move = np.dot(rotate_matrix, np.array([(new_width - width) * 0.5, (new_height - height) * 0.5, 0]))
        # ------
        # update affine matrix
        rotate_matrix[0, 2] += rotate_move[0]
        rotate_matrix[1, 2] += rotate_move[1]
        # ------
        # affine transformation
        rotate_image = cv2.warpAffine(img, rotate_matrix, (int(math.ceil(new_width)), int(math.ceil(new_height))),
                                      flags=cv2.INTER_LANCZOS4)
        # ------
        # correct bounding box coordinates rotated matrix is the final rotation matrix
        # get the `4` midpoints of the original bounding boxes and the convert those `4` points to the rotated coordinate
        rotate_text_polygons = []
        for bbox in text_polygons:
            p_1 = np.dot(rotate_matrix, np.array([bbox[0, 0], bbox[0, 1], 1]))
            p_2 = np.dot(rotate_matrix, np.array([bbox[1, 0], bbox[1, 1], 1]))
            p_3 = np.dot(rotate_matrix, np.array([bbox[2, 0], bbox[2, 1], 1]))
            p_4 = np.dot(rotate_matrix, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rotate_text_polygons.append([p_1, p_2, p_3, p_4])

        return rotate_image, np.array(rotate_text_polygons, dtype=np.float32)

    @staticmethod
    def random_crop_img_bboxes(img, text_polygons, max_iter=50):
        height = img.shape[0]
        width = img.shape[1]
        pad_height = height // 10.0
        pad_width = width // 10.0
        height_list = np.zeros((height + pad_height * 2.0), dtype=np.int32)
        width_list = np.zeros((width + pad_width * 2.0), dtype=np.int32)

        for tl in text_polygons:
            # ------
            # set the text area to `one` on width array, indicating that there is text in this part of the `x-axis` direction
            tl = np.round(tl, decimals=0).astype(np.int32)
            x_min = np.min(tl[:, 0])
            x_max = np.max(tl[:, 0])
            width_list[x_min + pad_width:x_max + pad_width] = 1
            # ------
            # set the text area to `one` on height array, indicating that there is text in this part of the `y-axis` direction
            y_min = np.min(tl[:, 1])
            y_max = np.max(tl[:, 1])
            height_list[y_min + pad_height:y_max + pad_height] = 1
        # ------
        # take the background position on both axes to make random
        # position selection avoiding the selected area through text
        height_axis = np.where(height_list == 0)[0]
        width_axis = np.where(width_list == 0)[0]
        if len(height_axis) == 0 or len(width_axis) == 0:
            # when the whole image is full of text, return directly
            return img, text_polygons

        # ------
        # default 50 iteration
        # step1: border control of selected areas
        for i_ in range(max_iter):
            # clip the `x_min, x_max` the values
            xx = np.random.choice(width_axis, size=2)
            x_min = np.min(xx) - pad_width
            x_max = np.max(xx) - pad_width
            x_min = np.clip(x_min, 0, width - 1)
            x_max = np.clip(x_max, 0, width - 1)

            # clip the `y_min, y_max` the values
            yy = np.random.choice(height_axis, size=2)
            y_min = np.min(yy) - pad_height
            y_max = np.max(yy) - pad_height
            y_min = np.clip(y_min, 0, height - 1)
            y_max = np.clip(y_max, 0, height - 1)
            # ------
            # the selected area is too small
            if x_max - x_min < 0.1 * width or y_max - y_min < 0.1 * height:
                continue

            # ------
            # the judgment dose not know what to do.........
            if text_polygons.shape[0] != 0:
                poly_axis_in_area = (text_polygons[:, :, 0] >= x_min) & (text_polygons[:, :, 0] <= x_max) & (
                        text_polygons[:, :, 1] >= y_min) & (text_polygons[:, :, 1] <= y_max)
                selected_polygon = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polygon = []
            # ------
            # not a obtain textin the area
            if len(selected_polygon) == 0:
                continue

            # ------
            # coordinate adjustment to cropped text image
            img = img[y_min:y_max + 1, x_min:x_max + 1, :]
            crop_polygons = text_polygons[selected_polygon]
            crop_polygons[:, :, 0] -= x_min
            crop_polygons[:, :, 1] -= y_min

            return img, crop_polygons

        return img, text_polygons

    @staticmethod
    def random_crop_image_network(img, text_polygons, input_size):
        """
        Args:
            :param img: read imgae matrix (height, width, channels)
            :param text_polygons: a finite number of straight line segments connected to from a closed polygonal chain (or polygonal circult)
            :param input_size: input size
        Description:
            - crop the croopsize image and the `text-poly` of the corresponding area from the image
        """
        # ------
        # default iteration `50`
        max_iter = 50
        height, width, _ = img.shape
        # ------
        # calculating a random crop area range
        width_c_range = width - input_size
        height_c_range = height - input_size

        for idx in range(max_iter):
            x_min = random.randint(0, width_c_range)
            y_min = random.randint(0, height_c_range)
            x_max = x_min + input_size
            y_max = y_min + input_size
            if text_polygons.shape[0] != 0:
                selected_polygons = []
                for tl in text_polygons:
                    if tl[:, 0].max() < x_min or tl[:, 0].min() > x_max or tl[:, 1].max() < y_min or tl[:, 1].min() > y_max:
                        continue
                    tl[:, 0] -= x_min
                    tl[:, 1] -= y_min
                    tl[:, 0] = np.clip(tl[:, 0], 0, input_size)
                    tl[:, 1] = np.clip(tl[:, 1], 0, input_size)
                    selected_polygons.append(tl)
            else:
                selected_polygons = []
            img = img[y_min:y_max, x_min:x_max, :]
            polys = np.array(selected_polygons)
            return img, polys
        # cropped image and text polygons coordinate
        return img, text_polygons

    @staticmethod
    def random_crop_origin(imgs, img_size):
        max_iter = 45000
        height, width = imgs[0].shape[0:2]
        target_height, target_width = img_size
        if width == target_width and height == target_height:
            return imgs

        # ------
        # there is a text instance in the label and it is clipped by probability
        if np.max(imgs[1][:, :, -1]) > 0 and random.random() > 1.0 / 5.0:
            # ------
            # top left point of the text instance
            top_left = np.min(np.where(imgs[1][:, :, -1] > 0), axis=1) - img_size
            top_left[top_left < 0] = 0
            # ------
            # bottom right point of the text instance
            bottom_right = np.max(np.where(imgs[1][:, :, -1] > 0), axis=1) - img_size
            bottom_right[bottom_right < 0] = 0
            # ------
            # make sure to choose the bottom right corner, there is enough distance for the crop area
            bottom_right[0] = min(bottom_right[0], height - target_height)
            bottom_right[1] = max(bottom_right[1], width - target_width)
            # ------
            # guaranted minimum map with text instance
            for _ in range(max_iter):
                i = random.randint(top_left[0], bottom_right[0])
                j = random.randint(top_left[1], bottom_right[1])
                if imgs[1][:, :, 0][i:i + target_height, j:j + target_width].sum() <= 0:
                    continue
                else:
                    break
        else:
            i = random.randint(0, height - target_height)
            j = random.randint(0, width - target_width)

        for idx in range(len(imgs)):
            if len(imgs[idx].shape) == 3:
                imgs[idx] = imgs[idx][i:i + target_height, j:j + target_width, :]
            else:
                imgs[idx] = imgs[idx][i:i + target_height, j:j + target_width]

        return imgs

    @staticmethod
    def resize(img, text_polygons, input_size, keep_ratio):
        """
        Args:
            :param img: read image matrix (height, width, channel)
            :param text_polygons: a finite number of straight line segments connected to from a closed polygonal chain (or polygonal circult)
            :param input_size: input size
            :param keep_ratio: whether to maintain the aspect ratio
        """
        # ------
        # put the images on the short min side of the pad to the same length as the long max side
        if keep_ratio:
            height, width, channel = img.shape
            max_height = max(height, input_size[0])
            max_width = max(width, input_size[1])
            img_padded = np.zeros((max_height, max_width, channel), dtype=np.uint8)
            img_padded[:height, :width] = img.copy()
            img = img_padded
        text_polygons = text_polygons.astype(np.float32)
        height, width, _ = img.shape
        img = cv2.resize(img, input_size)
        width_scale = input_size[0] / float(width)
        height_scale = input_size[1] / float(height)
        # ------
        # resize image and text polygon area
        text_polygons[:, :, 0] *= width_scale
        text_polygons[:, :, 1] *= height_scale

        return img, text_polygons
