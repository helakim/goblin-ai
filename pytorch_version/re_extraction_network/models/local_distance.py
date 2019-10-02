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
import torch


def hard_example_mining(distance_matrix, labels, return_inds=False):
    """
    Args:
        :param distance_matrix: convolution feature distance matrix
        :param labels: y values (positive or negative)
        :param return_inds: required by hard negative index
    Reference:
        - https://arxiv.org/pdf/1604.03540.pdf
    """
    assert len(distance_matrix.size()) == 2
    assert distance_matrix.size(0) == distance_matrix.size(1)
    num_batch = distance_matrix.size(0)

    positive_sample = labels.expand(num_batch, num_batch).eq(labels.expand(num_batch, num_batch).t())
    negative_sample = labels.expand(num_batch, num_batch).ne(labels.expand(num_batch, num_batch).t())

    # ------
    # distance_ap means (anchor, positive)
    # both `distance_ap` and `relative_n_inds` with shape [num_batch, 1]
    # indices of selected hard positive samples:
    # 0 <= 0 idx <= num_batch - 1
    distance_ap, relative_p_inds = torch.max(
        distance_matrix[positive_sample].contiguous().view(num_batch, -1),
        1,
        keepdim=True
    )
    # ------
    # distance_an means distance(anchor, negative)
    # both `distance_an` and `relative_n_index` with shape [num_batch, 1]
    # indices of selected hard negative samples:
    # 0 <= idx <= num_batch - 1
    distance_an, relative_n_inds = torch.min(
        distance_matrix[negative_sample].contiguous().view(num_batch, -1),
        1,
        keepdim=True
    )

    distance_ap = distance_ap.squeeze(1)
    distance_an = distance_an.squeeze(1)

    # ------
    # requirements by image queue search
    if return_inds:
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, num_batch).long())
               .unsqueeze(0).expand(num_batch, num_batch))

        # ------
        # positive index
        p_index = torch.gather(
            ind[positive_sample].contiguous().view(num_batch, -1),
            1,
            relative_p_inds
        )
        # ------
        # negative index
        n_index = torch.gather(
            ind[negative_sample].contiguous().view(num_batch, -1),
            1,
            relative_n_inds
        )
        p_index = p_index.squeeze(1)
        n_index = n_index.squeeze(1)

        return distance_ap, distance_an, p_index, n_index

    return distance_ap, distance_an