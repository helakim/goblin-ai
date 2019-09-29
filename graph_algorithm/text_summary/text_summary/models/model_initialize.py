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
from collections import Counter as CT


class TermPropertyInitialize(object):
    def __init__(self, candidate, tokens=list(), index=0):
        """
        :param index: Private key of sentence
        :param candidate: text
        :param tokens:

        Examples
        --------
            >>> import collections
            >>> str_tmp = list(['kt.kim', 'kt.kim', 'kt.kim', 'hela.kim','hela.kim'])
            >>> vote = collections.Counter(str_tmp)

            Counter({'kt.kim': 3, 'hela.kim': 2})
        """
        super(TermPropertyInitialize, self).__init__()
        self.index = index
        self.text = candidate
        self.tokens = tokens
        self.counter = CT(self.tokens)  # Count elements(tokens) from a dict(list)

    def __hash__(self):
        return hash(self.index)

    def __unicode__(self):
        return self.text

    def __repr__(self):
        try:
            return self.text.encoding(encoding='utf-8', errors=None)
        except:
            return str(self.text)
