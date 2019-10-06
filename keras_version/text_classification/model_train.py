# *********************************************************************
# @Project    text_classification
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
import os
import argparse

from preprocess.data import DataLoader, make_lookup_table, load_file

from models.sentencepiece import Graph
from models.embedding import W2vGraph
from models.gb_cnn import ConvolutionClassifier

from utils.chker import file_exist
from keras.callbacks import ReduceLROnPlateau


def hyper_parameter():
    args = argparse.ArgumentParser(description='Sentiment analysis model training hyperparameter')
    # ------
    # Text data parameter
    args.add_argument('-i', '--input_text', default='data/ratings.txt', type=str,
                      help='text location to make text only file')
    args.add_argument('-o', '--output_location', default='data/text_only.txt', type=str,
                      help="text only file location to save")
    args.add_argument('-max_len', '--max_len', default=30, type=int,
                      help="Parameter for training neural network to fix input sentnece len")
    # ------
    # Sentencepiece model parameter for train and save
    args.add_argument('-sp_text_path', default='data/text_only.txt', type=str, help="input text location")
    args.add_argument('-sp_model_path', default='pre_train/spm_model', type=str, help="trained model save location")
    args.add_argument('-sp_type', default='bpe', type=str,
                      help=" model type. Choose from bpe (default), unigram, char, or word."
                           "The input sentence must be pretokenized when using word type.")
    args.add_argument('-sp_vocab_size', default=37000, type=int, help="vocabulary size")
    args.add_argument('-sp_coverage', default=0.9995, type=float, help="amount of characters covered by the model")

    # ------
    # Word2vec model parameter for train and save
    args.add_argument('-w2v_model_path', default='pre_train/word2vec.model', type=str,
                      help="trained model save loaction")
    args.add_argument('-w2v_text_path', default='data/text_only.txt', type=str,
                      help="location of text to train word2vec")
    args.add_argument('-w2v_embed_size', default=300, type=int, help="word embedding size")
    args.add_argument('-w2v_type', default=1, type=int, help="0 for CBOW, 1 for skip-gram")
    args.add_argument('-w2v_loss', default=1, type=int, help="0 for negative sampling, 1 for hierarchical softmax")
    args.add_argument('-w2v_min_count', default=3, type=int, help="minimum count of words to train")
    args.add_argument('-w2v_window_size', default=2, type=int, help="window size of training model")

    # ------
    # Neural Net parameter for train and save
    args.add_argument('-nn_model_path', default='weights/', type=str, help='location of trained model to save')
    args.add_argument('-nn_model', default='cnn', type=str,
                      help='parameter to choose which model to train. list of model = [lstm, cnn]')
    args.add_argument('-nn_test_set_path', default='data/test_co_classification.txt')
    args.add_argument('-nn_train_set_path', default='data/train_co_classification.txt')
    args.add_argument('-nn_keep_prob', default=0.25, type=float)
    args.add_argument('-nn_filters', default=32, type=int)
    args.add_argument('-nn_train_type', default="w2v", type=str)
    args.add_argument('-nn_batch_size', default=128, type=int)
    args.add_argument('-nn_epoch_size', default=5, type=int)

    return args.parse_args()


def sp_worker(args):
    """ sentencepiece_worker is to train and save model
    """
    # ------
    # Data Loader to prepare training and saving 'sentence_piece' model
    DataLoader(data_load_path=args.input_text, data_save_path=args.output_location)

    # ------
    # Build model and training
    sp_model = Graph(
        text_path=args.sp_text_path,
        model_path=args.sp_model_path,
        type=args.sp_type,
        vocab_size=args.sp_vocab_size,
        coverage=args.sp_coverage
    )
    sp_model.op_train()
    file_exist(args.sp_model_path + ".model")
    file_exist(args.sp_text_path)


def w2v_worker(args, text_list):
    """ word2vec worker is to train and save model
    """
    # ------
    # Make indexed text
    padded_text = make_lookup_table(
        whole_text=text_list,
        text_list=text_list,
        vocab_size=args.sp_vocab_size,
        spm_model_path=args.sp_model_path,
        max_len=args.max_len
    )

    # ------
    # Train word2vec model and save
    w2v_model = W2vGraph(
        model_path=args.w2v_model_path,
        text_list=padded_text,
        embedding_size=args.w2v_embed_size,
        type=args.w2v_type,
        loss=args.w2v_loss,
        min_count=args.w2v_min_count,
        window_size=args.w2v_window_size
    )
    w2v_model.op_train()


def nn_train_worker(args):
    """ nn_train_worker is to train Neural Network and save model
    """
    # ------
    # Prepare train & test data
    whole_text, _ = load_file(data_load_path=args.input_text)
    train_text, train_label = load_file(data_load_path=args.nn_train_set_path)
    test_text, test_label = load_file(data_load_path=args.nn_test_set_path)
    # ------
    # stage.1: train dataset str -> idx
    train_text = make_lookup_table(
        whole_text=whole_text,
        text_list=train_text,
        vocab_size=args.sp_vocab_size,
        spm_model_path=args.sp_model_path,
        max_len=args.max_len
    )
    # ------
    # stage.2: test dataset str -> idx
    test_text = make_lookup_table(
        whole_text=whole_text,
        text_list=test_text,
        vocab_size=args.sp_vocab_size,
        spm_model_path=args.sp_model_path,
        max_len=args.max_len
    )
    # ------
    # initialization gb convoltuion neural network ()
    nn_model = ConvolutionClassifier(
        classes=2,
        vocab_size=args.sp_vocab_size,
        embed_size=args.w2v_embed_size,
        max_len=args.max_len,
        keep_prob=args.nn_keep_prob,
        filters=args.nn_filters,
        w2v_model_path=args.w2v_model_path,
        train_type=args.nn_train_type
    )

    nn_model_train = nn_model.build_graph()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=1e-5)
    nn_model_train.fit(train_text, train_label,
                       batch_size=args.nn_batch_size,
                       epochs=args.nn_epoch_size,
                       verbose=1,
                       validation_data=[test_text, test_label],
                       callbacks=[reduce_lr])

    nn_model_train.save(args.nn_model_path + args.nn_model + '.h5')


def main():
    # ------
    # training mode require
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # Get hyper parameters for other operations
    args = hyper_parameter()
    print("[@] Hyper parameter loaded: {}".format(args))

    # ------
    # Prepare dataset
    whole_text, _ = load_file(data_load_path=args.input_text)

    # ------
    # Train and save 'sentence_piece' model
    sp_worker(args)
    # print("[@] Sentencepiece csv file path {}, model file path {}")

    # ------
    # Training Word2vec model and save // prepare lookup_table
    w2v_worker(args, whole_text)
    # print("[@] Word2vec model saved")

    # ------
    # Network train
    nn_train_worker(args)


if __name__ == '__main__':
    main()
