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
import keras
import tqdm
import pandas as pd

import sentencepiece as spm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class DataLoader(object):

    def __init__(self, data_load_path, data_save_path):
        """
        :param data_load_path: text data location to load (eg. ./data/to/path)
        :param data_save_path: save location (eg. ./data/to/path)
        """
        super(DataLoader, self).__init__()
        # ------
        # instance variable
        self.data_load_path = data_load_path
        self.data_save_path = data_save_path
        self.columns = "document"

        # ------
        # instance static(local) function
        self.__path_checker()

    def __path_checker(self):
        try:
            open(self.data_load_path)
        except FileNotFoundError as e:
            print(e)

    def save_file(self, text_list):
        df = pd.DataFrame(columns=[self.columns])
        df[self.columns] = text_list
        df.to_csv(self.data_save_path, index=False, header=False)


def make_lookup_table(whole_text, text_list, vocab_size, spm_model_path, max_len):
    """
    """
    # ------
    # Sentencepiece Load
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_path + ".model")

    # ------
    # Put every word in on bag
    word_bag = []
    for sentence in tqdm.tqdm(whole_text, desc='Word splitting: '):
        for word in sentence:
            word = sp.EncodeAsPieces(word)
            word_bag.append(word)

    # ------
    # Making Lookup_table
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(word_bag)

    # ------
    # Indexing and padding input text
    text_list = [sp.EncodeAsPieces(str(i)) for i in text_list]
    indexed_text = tokenizer.texts_to_sequences(text_list)
    padded_text = pad_sequences(indexed_text, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0.0)

    return padded_text


def load_file(data_load_path, seperator="\t", columns="document"):
    """
    Args:
        :param seperator: delimiter for columns split

    TODO:
        - raw_data key value to be flexible
        - key error check
    """
    raw_data = pd.read_csv(data_load_path, sep=seperator)
    text_bucket = raw_data[columns]
    text_bucket = [str(i).split() for i in text_bucket]
    label_bucket = raw_data["label"]

    return text_bucket, keras.utils.to_categorical(label_bucket)
