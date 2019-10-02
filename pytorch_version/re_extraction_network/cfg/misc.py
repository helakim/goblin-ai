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
import numpy as np


def _traceback_ops(x):
    i, j = np.array(x.shape) - 1
    p, q = [i], [j]

    while (i > 0) or (j > 0):
        trace_back = np.argmin((x[i, j - 1], x[i - 1, j]))
        if trace_back == 0:
            j -= 1
        else:
            i -= 1
        p.insert(0, i)
        q.insert(0, j)

    return np.array(p), np.array(q)


def distance_matrix_weight(matrix):
    m, n = matrix.shape[:2]
    distance = np.zeros_like(matrix)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                distance[i, j] = matrix[i, j]
            elif (i == 0) and (j > 0):
                distance[i, j] = distance[i, j - 1] + matrix[i, j]
            elif (i > 0) and (j == 0):
                distance[i, j] = distance[i - 1, j] + matrix[i, j]
            else:
                distance[i, j] = np.min(np.stack([distance[i - 1, j], distance[i, j - 1]], axis=0), axis=0) + matrix[
                    i, j]
    path = _traceback_ops(distance)

    return distance[-1, -1] / sum(distance.shape), distance, path
