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
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Conv1D, MaxPool1D
from keras.layers.embeddings import Embedding
from keras.layers.core import Dropout
from gensim.models import Word2Vec


def top1_acc(x, y):
    return keras.metrics.top_k_categorical_accuracy(x, y, k=1)


class ConvolutionClassifier(object):
    def __init__(self, classes, vocab_size, embed_size, max_len, keep_prob, filters, w2v_model_path, train_type):
        """
        Args:
            :param classes: number of classifier units
            :param vocab_size: number of word
            :param embed_size: dimensions of vectors spaces
            :param max_len: max len
            :param keep_prob: threshold of dropout layers
            :param filters: convolution filters
            :param w2v_model_path: word to vectors path
            :param train_type: static or non static
        """
        super(ConvolutionClassifier, self).__init__()
        self.classes = classes
        self.vocab_size = vocab_size
        self.dim = embed_size
        self.max_len = max_len
        self.keep_prob = keep_prob
        self.filters = filters
        self.w2v_model_path = w2v_model_path
        self.train_type = train_type

    def build_graph(self):
        if self.train_type == "rand":
            weights = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.dim))
        else:
            # pretrained word vectors
            model = Word2Vec.load(self.w2v_model_path)
            weights = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.dim))
            for i in list(model.wv.vocab):
                weights[int(i)] = model.wv.get_vector(i)

        input_shape = Input((self.max_len,))
        # -----------
        # embedding layers
        embedding = Embedding(input_dim=self.vocab_size, output_dim=self.dim, weights=[weights], trainable=True)(input_shape)
        embedded_output = Dropout(self.keep_prob)(embedding)

        # -----------
        # convolution layers_0
        conv_0 = Conv1D(self.filters, 3, strides=1, kernel_initializer='he_normal', padding='valid', activation='relu')(embedded_output)
        max_pool_0 = MaxPool1D(2)(conv_0)
        output_0 = Flatten()(max_pool_0)

        # -----------
        # convolution layers_1
        conv_1 = Conv1D(self.filters, 4, strides=1, kernel_initializer="he_normal", padding="valid", activation="relu")(embedded_output)
        max_pool_1 = MaxPool1D(2)(conv_1)
        output_1 = Flatten()(max_pool_1)

        # -----------
        # convolution layers_2
        conv_2 = Conv1D(self.filters, 5, strides=1, kernel_initializer="he_normal", padding="valid", activation="relu")(embedded_output)
        max_pool_2 = MaxPool1D(2)(conv_2)
        output_2 = Flatten()(max_pool_2)
        # -----------
        # Concat
        concatenated_max_pool = keras.layers.Concatenate(axis=-1)([output_0, output_1, output_2])
        drop_out = Dropout(self.keep_prob)(concatenated_max_pool)

        # -----------
        # Fully Connected Layers
        outputs = Dense(2, activation="softmax")(drop_out)
        model = Model(inputs=input_shape, outputs=outputs)
        ops = keras.optimizers.Adam(1e-3)

        model.compile(loss="categorical_crossentropy", optimizer=ops, metrics=['acc'])
        model.summary()

        return model
