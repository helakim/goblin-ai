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
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from logger import logger as cerberus


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
    cerberus.info('[@] model saved to {}'.format(checkpoint_path))


def load_checkpoint(checkpoint_path, model, device, optimizer=None):
    model_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(model_state_dict['state_dict'])
    if optimizer is None:
        optimizer.load_state_dict(model_state_dict['optimizer'])
    state_epoch = model_state_dict['epoch']
    cerberus.info('[@] model loaded from {}'.format(checkpoint_path))

    return state_epoch


def elapsed_time(func):
    def new_func(*args, **args2):
        start_time = time.time()
        backward = func(*args, *args2)
        cerberus.info('[@] {} elpased time {:.3f}(s)'.format(func.__name__, time.time() - start_time))
        return backward

    return new_func


def model_weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def cuda_tensor_convert(device):
    """
    Args:
        :param device: number of machine
    """
    if torch.cuda.is_available():
        if type(device) == tuple or type(device) == list:
            # ------
            # Multi GPU
            return [x.to(cpu_gpu_check()) for x in device]
        return device.to(cpu_gpu_check())
    return device


def cpu_gpu_check():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


def numpy_to_tensor(x, use_cuda=True):
    """
    Args:
        :param x: numpy array for image
        :param use_cuda: use GPU
    """
    if x.dtype == np.uint:
        image = x.astype(np.float32)
    else:
        assert x.dtype == np.float32
    x = np.rollaxis(x, axis=2, start=0)
    x = x[None, :, :, :]
    x = torch.from_numpy(x)
    if use_cuda:
        x = cuda_tensor_convert(x)

    return x


def prediction_coords(location, priors_bbox, variances):
    """
    Args:
        :param location: location predictions for locations layers [regression layers]
        :param priors_bbox: coordinates in center offset [suggest center x, y]
        :param variances: variances of prior bounding boxes

    Testing: True
    """
    tensor = (priors_bbox[:, :2] + location[:, :2] * variances[0] * priors_bbox[:, 2:],
              priors_bbox[:, 2:] * torch.exp(location[:, 2:] * variances[1]))
    coords = torch.cat(tensor, dim=1)
    # min_x, min_y
    coords[:, :2] -= coords[:, 2:] / 2
    # max_x, max_y
    coords[:, 2:] += coords[:, :2]

    return coords  # [(x_min, y_min, x_max, y_max)]


def show_masking_bbox_v1(x, anchors):
    alpha_value = 0.671
    img = x
    for anchor in anchors:
        x_over = x.copy()
        x_min, y_min, x_max, y_max = [np.int(_) for _ in anchor]
        cv2.rectangle(x, (x_min, y_min), (x_max, y_max), (255, 255, 0), -1)
        cv2.addWeighted(x_over, alpha_value, x, 1 - alpha_value, 0, x)
        cv2.putText(img, 'face', (x_min, y_min -5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1)
        cv2.rectangle(x, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
