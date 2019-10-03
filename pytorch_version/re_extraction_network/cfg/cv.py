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
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tqdm
from PIL import Image
from cfg.misc import distance_matrix_weight


def gray_scale(image):
    image = torch.sum(image, dim=0)
    image = torch.div(image, image.shape[0])

    return image


def layer_output(x, module_lists, vis):
    outputs = []
    names = []
    outputs_im = []
    for block_name, cnn_block in tqdm.tqdm(module_lists, desc='feature extraction: '):
        x = cnn_block(x)
        outputs.append(x)
        names.append(str(cnn_block))

    for im in outputs:
        im = im.squeeze(0)
        img_normalized = gray_scale(im)
        outputs_im.append(img_normalized.data.cpu().numpy())

    for im in tqdm.tqdm(range(len(outputs_im)), desc='visdom inference: '):
        plt.imshow(outputs_im[im], interpolation='nearest', aspect='auto')
        plt.title('layer_stage_{}'.format(str(im)), fontsize=15)
        vis.matplot(plt)

    return outputs_im[1]


def read_image(image_path):
    flag = False
    if not os.path.exists(image_path):
        raise IOError('[@] image does not exist: {}'.format(image_path))
    while not flag:
        try:
            img = Image.open(image_path).convert('RGB')
            flag = True
        except IOError:
            print('[@] Exception incurred when reading: {}'.format(image_path))
            pass

    return img


def img_to_tensor(img, transform):
    img = transform(img)
    img = img.unsqueeze(0)

    return img


def show_convolution_feature(x, feature_channel=64, color_map='jet'):
    for j_ in range(len(x)):
        for i_ in range(len(feature_channel)):
            ax = plt.subplot(4, 16, i_ + 1)
            ax.set_title('number: {}'.format(i_))
            ax.axis('off')
            plt.imshow(x[j_].cpu().data.numpy()[0, i_, :, :])
        plt.show()


def feature_flatten(x: np.ndarray):
    input_shape = x.shape
    x = x.reshape(input_shape[0] * input_shape[1], input_shape[2])

    return x


def merged_feature(feature_list, input_shape, sample_rate=None):
    def per_process(torch_feature_map):
        np_feature_map = torch_feature_map.cpu().data.numpy()[0]
        np_feature_map = np_feature_map.transpose(1, 2, 0)
        input_shape = np_feature_map.shape[:2]

        return np_feature_map, input_shape

    def resize_as(torch_feature_map, input_shape):
        np_feature_map, input_shape_2 = per_process(torch_feature_map)
        scale = input_shape[0] / input_shape_2[0]
        np_feature_map_1 = np_feature_map.repeat(scale, axis=0).repeat(scale, axis=1)

        return np_feature_map_1

    feature_map = resize_as(feature_list[0], input_shape)
    for idx in range(1, len(feature_list)):
        feature_map_flush = resize_as(feature_list[idx, input_shape])
        feature_map = np.concatenate((feature_map, feature_map_flush), axis=-1)
    if sample_rate > 0:
        feature_map = feature_map[0:-1:sample_rate, 0:-1, sample_rate, :]

    return feature_map


def show_distance_diff(local_image_path, image_path, similarity, bbox, output_shape=(64, 128)):
    source_img = cv2.imread(local_image_path)
    target_img = cv2.imread(image_path)

    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    source_img = cv2.resize(source_img, output_shape)
    target_img = cv2.resize(target_img, output_shape)

    cv2.rectangle(source_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(0, 255, 0),
                  thickness=1)
    prob = np.where(similarity == np.max(similarity))

    y, x = prob[0][0], prob[1][0]
    cv2.rectangle(target_img, (x - bbox[2] / 2, y - bbox[3] / 2), (x + bbox[2] / 2, y + bbox[3] / 2), color=(0, 255, 0),
                  thickness=1)

    plt.subplot(1, 3, 1).set_title('image_patch')
    plt.imshow(source_img)

    plt.subplot(1, 3, 2).set_title('max similarity: {}'.format(str(np.max(similarity))))
    plt.imshow(target_img)

    plt.subplot(1, 3, 3).set_title('similarity')
    plt.imshow(similarity)


def img_processing(img, c=cv2.COLOR_BGR2RGB, r=(64, 128)):
    if isinstance(img, str):
        img = cv2.imread(img)
    img = cv2.cvtColor(img, c)
    img = cv2.resize(img, dsize=r)

    return img


def tensor_to_vector(local_image_path, image_path, distance, c=(255, 255, 0), vis=None):
    def _drew_line(img, similarity):
        for i in range(1, len(similarity)):
            cv2.line(img, (0, i * 16), (63, i * 16), c)
            cv2.line(img, (96, i * 16), (160, i * 16), c)

    def _drew_path(img, path):
        for i in range(len(path[0])):
            cv2.line(img, (64, 8 + 16 * path[0][i]), (96, 8 + 16 * path[1][i]), (255, 255, 255))

    source_img = img_processing(local_image_path)
    target_img = img_processing(image_path)

    img = np.zeros((128, 160, 3)).astype(source_img.dtype)
    img[:, :64, :] = source_img
    img[:, -64:, :] = target_img
    _drew_line(img, distance)
    div_dist, all_dist, split_path = distance_matrix_weight(distance)
    origin_distance = np.mean(np.diag(distance))
    _drew_path(img, split_path)

    # right image
    plt.subplot(1, 2, 1).set_title(
        'feature distance: {:.4f} \n origin distance: {:.4f}'.format(div_dist, origin_distance))
    plt.subplot(1, 2, 1).set_xlabel('feature result')
    plt.imshow(img, interpolation='nearest', aspect='auto')

    plt.subplot(1, 2, 2).set_title('distance map')
    plt.subplot(1, 2, 2).set_xlabel('right image')
    plt.subplot(1, 2, 2).set_ylabel('left image')
    plt.imshow(distance, interpolation='nearest', aspect='auto')

    plt.subplots_adjust(bottom=0.1, left=0.075, right=0.85, top=0.9)
    cax = plt.axes([0.9, 0.25, 0.025, 0.5])
    plt.colorbar(cax=cax)
    if vis is not None:
        vis_feature_img = cv2.resize(img, (680, 680))
        vis.image(vis_feature_img.transpose((2, 0, 1)), opts=dict(
            title='feature distance: {:.4f}, origin distance: {:.4f}'.format(div_dist, origin_distance),
            jpgquality=100
        ))
        vis.matplot(plt)
    else:
        plt.show()

    return div_dist, origin_distance
