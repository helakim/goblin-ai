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
import sentencepiece as spm


class Graph(object):
    """
    """

    def __init__(self, text_path, model_path, type, vocab_size, coverage):
        """
        Args:
            :param text_path: text input path
            :param model_path: saved model path
            :param type: model type. Choose from bpe (default), uni, char, or word. The input sentence must
                        be pretokenized when using word type.
            :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
            :param coverage:  amount of characters covered by the model

        Description:
            - SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text
             generation systems where the vocabulary size is predetermined prior to the neural model training.

        Reference:
            - Neural Machine Translation of Rare Words with Subword Units
                - http://www.aclweb.org/anthology/P16-1162

            - Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates
                - https://arxiv.org/abs/1804.10959

            - https://github.com/google/sentencepiece
        """
        super(Graph, self).__init__()
        # ------
        # instance variable
        self.text_path = text_path
        self.model_path = model_path
        self.type = type
        self.vocab_size = vocab_size
        self.coverage = coverage

        # ------
        # instance local operator
        """ SentencePieceProcessor
        Example:
            >>> import sentencepiece as spm
            >>> 링크 참조 
            >>> spm_train --input=<input> --model_prefix=<model_name> --vocab_size=8000 --character_coverage=1.0 --model_type=<type>
        """
        self.sp = spm.SentencePieceProcessor()

    def op_train(self):
        spm.SentencePieceTrainer.Train(
            ' --input={} --model_type={} --model_prefix={} --vocab_size={} --input_sentence_size=10000000'
            ' --character_coverage={}'.format(self.text_path, self.type, self.model_path, self.vocab_size, self.coverage)
        )
