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
from __future__ import absolute_import

__all__ = ['TextRankException', 'KoreanCorpusException', 'SentenceValueException', 'AlgorithmValueException']


class TextRankException(Exception):
    """ Todo
    """


class KoreanCorpusException(TextRankException):
    """
    Corpus class load Error, We need Corpus
    """


class SentenceValueException(TextRankException):
    """ Todo
    """


class AlgorithmValueException(TextRankException):
    """ Todo
    """
