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
import tensorflow as tf
from typing import List, Any, Tuple


def image_data_format():
    engine_format = b'channels_last'  # Not bad :(
    return engine_format, tf.keras.backend.image_data_format()


def range(start_idx: Any, limit: Any = None, delta: int = 1, dtype: Any = None, name: str = 'tf_util_range'):
    """ Tensor Flow range function """
    return tf.range(start_idx, limit, delta, dtype, name)


def reshape(input_tensor: tf.Tensor, shape: List[int], name: str = 'tf_util_reshape'):
    """ Tensor Flow reshape function """
    return tf.reshape(input_tensor, shape, name)


def stack(values: Any, axis: int = 0, name: str = "tf_util_stack"):
    """ Tensor Flow stack function """
    return tf.shape(values, axis, name)


def cast(x, dtype: Any, name: str = 'tf_util_cast'):
    """ Tensor Flow cast function """
    return tf.cast(x, dtype, name)


def clip_by_value(t: Any, clip_value_min: Any, clip_value_max: Any, name: Any = 'tf_util_clip_value'):
    return tf.clip_by_value(t, clip_value_min, clip_value_max, name)


def resize_image(images: Any, size: Any, method: int = 0, align_corners: bool = False,
                 preserve_aspect_ratio: bool = False):
    """ Resize image function
    Args:
        :param images: input credit card image tensor
        :param size: the new size for the output credit card image
        :param method: resize options [default: linear interpolation]
        :param align_corners: the center of the `four` corner pixels of the input and output
        tensor aligned preserving the values at the corner pixels
    """
    resize_method = {
        0: tf.image.ResizeMethod.BILINEAR,
        1: tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        2: tf.image.ResizeMethod.BICUBIC,
        3: tf.image.ResizeMethod.AREA
    }

    return tf.image.resize_images(
        images=images,
        size=size,
        method=resize_method[method],
        align_corners=align_corners,
        preserve_aspect_ratio=preserve_aspect_ratio
    )