# *********************************************************************
# @Project    goblin-ai
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   06/10/2019
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
import random
import numpy as np
from sklearn.model_selection import train_test_split, KFold


def batch_provider(data, batch_size, num_epochs, perm=True):
    # ------
    # convert to train and validation samples
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch_ in range(num_epochs):
        if perm:
            # Randomly permute a sequence or range
            # if `x` is a multi-dimensional array, it is only shuffled along it's first index
            perm_length = np.arange(data_size)
            perm_indices = np.random.permutation(perm_length)
            permed_data = data[perm_indices]
        else:
            permed_data = data
        for batch_num in range(num_batches_per_epoch):
            s_idx = batch_num * batch_size  # First mini-batch index
            e_idx = min((batch_num + 1) * batch_size, data_size)  # Last mini-batch index

            yield permed_data[s_idx: e_idx]


def train_test_spliter(x, targets, ratio=0.1):
    """ Split n-arrays into random train and validation subsets
    Args:
        :param x: Allowed inputs are lists, numpy arrays (eg. text dataset)
    """
    x_train, x_test, y_train, y_test = train_test_split(x, targets, test_size=ratio)
    return (x_train, x_test, y_train, y_test)


def k_folder_validator(x, targets, k=4, perm=True):
    """ K-Folds cross validator
    Example
        >>> x -> train feature
        >>> targets -> labels

        >>> for train_, test_ in KFold.split(x)
                x_train, x_test = x[train_],  x[test_]
                y_train, y_test = targets[train_], targets[test_]
    """
    return KFold(n_splits=k, random_state=None, shuffle=perm)


def average_precision_score(y, y_pred):
    """ Compute average precision from prediction scores
    """
    pass


def precision_matrix(y, y_pred):
    """
    """
    pass


def roc_auc_matrix(y, y_score):
    """ Compute Area Under the Receiver Operating Characteristic Curve
    """
    pass


def f1_matrix(y, y_pred):
    """ Compute the `F1` score also know as blanced `F-score` or `F-measure`
    2 * (precision * recall) / (precision + recall)
    """
    pass

