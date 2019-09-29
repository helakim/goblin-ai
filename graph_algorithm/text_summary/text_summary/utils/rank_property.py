# *********************************************************************
# @Project    text_summary
# @FILE       kyung_tae_kim (hela.kim)
# @Copyright: hela
# @Created:   29/09/2019
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
from ..utils.pos_tags import MECAB_KO_USEFUL_TAGS, TWITTER_KO_USEFUL_TAGS
from text_summary.utils.word import KOREAN_STOP_WORDS

__all__ = ['RankProperty']


class RankProperty(object):
    def __init__(self):
        super(RankProperty, self).__init__()
        self.decay_constant = dict(window=10, alpha=0.25)
        self.korean_corpus = dict(worker='Twitter', tags=TWITTER_KO_USEFUL_TAGS, cluster='DENSITY_BASED_ALGORITHM',
                                  algorithm='J', splitter=list(['. ', '\n', '.\n']))
        self.word = dict(
            non_below=2,
            non_above=0.85,
            token_min_length=2,
            stop=KOREAN_STOP_WORDS
        )

        self.hyper_parameter = dict(
            min_cluster=2,
            max_dic_length=None,
            threshold=0.85,
            matrix_convex=False,
            compacted=True
        )

    @property
    def get_decay_constant(self):
        return self.decay_constant

    @property
    def get_korean_corpus(self):
        return self.korean_corpus

    @property
    def get_word(self):
        return self.word

    @property
    def get_hyper_parameter(self):
        return self.hyper_parameter

    def __repr__(self):
        property_tmp = self.__class__.__name__
        return property_tmp
