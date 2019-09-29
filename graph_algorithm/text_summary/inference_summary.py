# *********************************************************************
# @Project    goblin-ai
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
import os
import codecs
import argparse
import operator
from pprint import PrettyPrinter
from pprint import pprint
from text_summary.models.summary_algorithm import CerberusSummary
from text_summary.cfg.logger import logger as say

parser = argparse.ArgumentParser(description='cerberus Summary :)')
parser.add_argument('--type', dest='data_type', default='file', type=str)
args = parser.parse_args()
TEXT_SUMMARY = object()


def type_checker():
    text_bucket = list()

    if operator.__eq__(args.data_type, 'list'):
        """
        Example
        -------
        >>> text_bucket = list([
                '구급대는 아이 얼굴과 온몸에서 타박상과 멍 자국을 발견하고 경찰에 신고했습니다',
                '아동 학대를 의심한 경찰은 부모를 조사한 끝에 의붓아버지 26살 A 씨를 긴급체포했습니다',
                '혐의를 부인했던 A 씨는 계속된 추궁에 아이를 20시간 넘게 때린 사실을 인정했습니다',
                '특히 끔찍한 폭행은 가족들이 모두 있는 자리에서 이뤄졌습니다',
        """
        text_bucket = list()
    elif operator.__eq__(args.data_type, 'file'):
        FILE_DIR = os.path.join(os.getcwd(), 'samples/news_2.txt')
        if not os.path.exists(FILE_DIR):
            raise FileNotFoundError('File not found exception')

        with codecs.open(FILE_DIR, mode='r', encoding='utf-8', errors='strict', buffering=1) as f:
            for (_, sentence) in enumerate(f):
                text_bucket.append(sentence)
    else:
        PrettyPrinter(indent=4)

        raise ValueError('ArgumentParser Error')

    return text_bucket


def summary_loader(sentences_bucket):
    global TEXT_SUMMARY
    sentence_status = TEXT_SUMMARY.sentence_processing(sentences_bucket)
    if sentence_status:
        # Target sentence
        say.debug(sentence_status)
        sentences_concat = TEXT_SUMMARY.summary_works()

        for _, sentence in enumerate(sentences_concat):
            say.info("text: {}".format(sentence[0]))
            # say.info("tokens: {}".format(sentence[1]))
            # say.info("counter: {}".format(sentence[2]))
        return sentences_concat
    else:
        # Origin sentence
        pprint('origin text: {}'.format(sentences_bucket), depth=1, width=60)
        return sentences_bucket


if __name__ == '__main__':
    sentences_bucket = type_checker()
    if type(sentences_bucket) is list or sentences_bucket:
        TEXT_SUMMARY = CerberusSummary()
        summary_result = summary_loader(sentences_bucket)
    else:
        raise ValueError('Cluster branch tries the sequence if it is not clustered or empty')
