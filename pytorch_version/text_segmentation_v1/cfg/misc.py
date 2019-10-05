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
import math
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import glob
import natsort
from collections import OrderedDict


def display_image(imgs, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    for img in imgs:
        plt.figure(figsize=(32, 32))
        plt.imshow(img, cmap=None if color else 'pink')
        plt.colorbar()


def display_multiple_image(imgs, color=False):
    if (len(imgs.shape) == 3 and color) or (len(imgs.shape) == 2 and not color):
        imgs = np.expand_dims(imgs, axis=0)
    fig = plt.figure(figsize=(32, 32))
    columns = 3
    rows = 2
    for idx in range(1, columns * rows + 1):
        img = imgs[idx - 1]
        fig.add_subplot(rows, columns, idx)
        plt.imshow(img, cmap=None if color else 'pink')
        plt.colorbar()


def draw_bbox(img_path, result, color=(255, 0, 255), thickness=3):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    # ------
    # TODO.md: add draw polygons function
    for p in result:
        p = p.astype(np.int)
        cv2.line(img_path, tuple(p[0]), tuple(p[1]), color=color, thickness=thickness)
        cv2.line(img_path, tuple(p[1]), tuple(p[2]), color=color, thickness=thickness)
        cv2.line(img_path, tuple(p[2]), tuple(p[3]), color=color, thickness=thickness)
        cv2.line(img_path, tuple(p[3]), tuple(p[0]), color=color, thickness=thickness)

    return img_path


def save_checkpoint(checkpoint_path, model, optimizer, epoch):
    model_state_dict = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(model_state_dict, checkpoint_path)
    print('[@] model saved to {}'.format(checkpoint_path))


def load_checkpoint(checkpoint_path, model, device, optimizer=None):
    model_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_state_dict['state_dict'])
    if optimizer is None:
        optimizer.load_state_dict(model_state_dict['optimizer'])
    state_epoch = model_state_dict['epoch']
    print('[@] model loaded from {}'.format(checkpoint_path))

    return state_epoch


def elapsed_time(func):
    def new_func(*args, **args2):
        start_time = time.time()
        backward = func(*args, *args2)
        print('[@] {} elapsed time {:.3f}(s)'.format(func.__name__, time.time() - start_time))
        return backward

    return new_func


def model_weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def cached_state_model_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def bbox_generator(text_map, link_map, text_threshold, link_threshold, low_text):
    # prepare data
    link_map, text_map = link_map.copy(), text_map.copy()
    img_h, img_w = text_map.shape

    """ labeling method """
    ret, text_score = cv2.threshold(text_map, low_text, 1, 0)
    ret, link_score = cv2.threshold(link_map, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        # ------
        # bbox too smaller
        if size < 10:
            continue

        if np.max(text_map[labels == k]) < text_threshold:
            continue

        # ------
        # suggest segment areas
        seg_maps = np.zeros(text_map.shape, dtype=np.uint8)
        seg_maps[labels == k] = 255
        seg_maps[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        seg_maps[sy:ey, sx:ex] = cv2.dilate(seg_maps[sy:ey, sx:ex], kernel)

        np_contours = np.roll(np.array(np.where(seg_maps != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        start_idx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_idx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def detection_anchors_generator(y, text_threshold=0.56, link_threshold=0.4122139, low_text=0.41338871 + 3e-6):
    """
    Args:
        :param y: `y hat`
        :param text_threshold: text region of regression score
        :param link_threshold: link region of regression score
    """
    t_map, l_map = y[0,:,:,0].cpu().data.numpy(), y[0,:,:,1].cpu().data.numpy()
    boxes, labels, mapper = bbox_generator(t_map, l_map, text_threshold, link_threshold, low_text)
    # ------
    # TODO add polygons coordinate
    polygons = [None] * len(boxes)

    return boxes, polygons


def clip_xxyy(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def image_file_sorted(input_images_path=None, ext='jpg'):
    image_files = natsort.natsorted(glob.glob(input_images_path + '/*.{}'.format(ext)))

    return image_files