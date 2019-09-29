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
import operator
import math
import networkx
import re
import numpy as np

from sklearn.cluster import DBSCAN
from gensim.corpora import TextCorpus, Dictionary
from konlpy import tag as corpus
from text_summary.cfg.logger import logger as say
from text_summary.exceptions.custom_exception import *

# Todo: Refactor keyword_stop_list file (Config)
SENTENCES_MAX_LENGTH = 7


class KoreanSentenceWorker(object):
    def __init__(self, korean_corpus_worker, korean_corpus_splitter, token_min, korean_stop_words, **kwargs):
        # Complete
        try:
            # -------------
            # Twitter
            self.tagger = corpus.Okt()
            # self.tagger_options = dict(stem=bool(kwargs.get('stem', True)))
            # -------------
            # Mecab
            # self.tagger = corpus.Mecab()
            # self.tagger_options = dict(flatten=bool(kwargs.get('flatten', True)))
        except:
            say.error('Could not load korean corpus')
            raise KoreanCorpusException(TextRankException('Could not load korean corpus'))
        self.korean_corpus_tags = korean_corpus_worker
        self.delimiters = korean_corpus_splitter
        self.min_token_length = token_min
        self.keyword_stop_lists = korean_stop_words

        self.sentence_splitter = self.__sentences_splitter()
        # Configurations Korean Corpus
        # Option [1. mecab, 2. twitter]
        # self.pos = lambda sentence: self.tagger.pos(sentence, **self.tagger_options)
        self.pos = lambda sentence: self.tagger.pos(sentence, stem=False)

    def __sentences_splitter(self):
        """ Escape all the characters in pattern except ASCII letters, numbers and '_'.
        """
        escaped_delimiters = '|'.join([re.escape(delimiter) for delimiter in self.delimiters])

        return lambda value: re.split(escaped_delimiters, value)

    def __text_to_tokens(self, text):
        tokens = []
        word_tag_pairs = self.pos(text)
        for word, tag in word_tag_pairs:
            if word in self.keyword_stop_lists:
                continue

            if tag not in self.korean_corpus_tags:
                continue

            tokens.append("word: {} tags: {}".format(str(word), str(tag)))
        return tokens

    def text_to_sentences(self, text):
        from text_summary.models.model_initialize import TermPropertyInitialize
        if type(text) is list:
            """List check (document)
            """
            document = text
            text = ''
            for _ in range(len(document)):
                text += document[_] + '\n'
        candidates = self.sentence_splitter(text.strip())
        sentences = []
        index = 0
        for candidate in candidates:
            while len(candidate) and (candidate[-1] == '.' or
                                      candidate[-1] == '\n' or
                                      candidate[-1] == '.\n'):
                candidate = candidate.strip(' ').strip('.')
            if not candidate: continue
            tokens = self.__text_to_tokens(candidate)
            if len(tokens) < 2:  continue  # Token length
            sentence = TermPropertyInitialize(candidate, tokens, index)
            sentences.append(sentence)
            index += 1
        return sentences


class KoreanSentenceCorpus(TextCorpus):
    def __init__(self, sentences, no_below, no_above, max_size=None):
        """2048
                ::math Bag of word::
                      Suppose the document collection has `n` distinct words,
                      `w(1) , ... , w(n)`
                      Each document is characterized by an `n-dimensional`
                      vector whose `i(th)` component is the frequency of the word
                      `w(i) in the document
                :class TextCorpus:
                    - Simplify the pipeline of getting Bag of words vectors from plain text
                :param dictionary:
                    - class Dictionary
                :param metadata:
                    - Yield metadata with each document(default True)
                :param input:
                    - Corpus documents(sentence)

                Examples
                --------
                    >>> from gensim.corpora import TextCorpus
                    >>> KoreanSentenceCorpus(TextCorpus)
                    >>>     stop_keyword = frozenset(["anything"])
                    >>>     korean_keyword(stop_keyword)
                    >>>     def get_texts(self):
                    >>>         for doc in self.documents():
                    >>>             yield [doc.tokens]
        """
        super(KoreanSentenceCorpus, self).__init__()
        self.metadata = False
        self.sentences = sentences

        # Construct word <=> id(index) mapping
        self.dictionary = Dictionary(documents=self.get_texts(), prune_at=max_size)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=max_size)
        self.dictionary.compactify()
        self.bows = [self.dictionary.doc2bow(tokens) for tokens in self.get_texts()]

    def get_texts(self):
        for sentence in self.sentences:
            yield sentence.tokens

    def __len__(self):
        self.length = sum(1 for _ in self.get_texts())
        return self.length


class CerberusSummary(object):
    """
    ::math Jaccard Index::
        - The Jaccard distance, which measures dissimilarity between sample sets,
        is complementary to the markov Index coefficient and is obtained by subtracting
        the markov coefficient from `1` & equivalently, by dividing the difference of
        the sizes of the union and the intersection of two sets by the size of the union

                                              | A ∩ B |
        J(A, B) = |A U B| / | |A N B| =  --------------------
                                         |A| + |B| - |A ∩ B|
        Example
        -------
        Intersection of union = Area of Union / Area of Overlap
    """

    def __init__(self, **kwargs):
        super(CerberusSummary, self).__init__()
        from text_summary.utils.rank_property import RankProperty as P
        el = P()

        self.decay_window = el.get_decay_constant['window']
        self.decay_alpha = el.get_decay_constant['alpha']
        self.uniform_similarity = self.__similarity_jaccard_algorithm  # uniform
        self.multi_sets_sim = lambda st1, st2: self.__exponential_decay(st1, st2) * self.uniform_similarity(st1, st2)
        """
         refactor function name of decay
          - factory (sentences)
        """
        ######################################################
        #                  Class property(Local)             #
        ######################################################
        korean_corpus_worker = el.get_korean_corpus['worker']
        korean_corpus_tags = el.get_korean_corpus['tags']
        korean_corpus_splitter = el.get_korean_corpus['splitter']
        token_min = el.get_word['token_min_length']
        korean_stop_words = el.get_word['stop']
        cluster_algorithm = el.get_korean_corpus['cluster']

        ######################################################
        #                    Algorithm(K-Cluster)            #
        ######################################################
        if cluster_algorithm == 'DENSITY_BASED_ALGORITHM':
            """ Algorithm
                - Density Based Spatial Clustering of Application with noise
                To find a cluster Density-Based Spatial Clustering of Applications with noise
                starts with an arbitrary point `p` and retrieves all point s density-reachable from `p`
                wrt. `Eps` and `MinPts` if `p` is a core point, this procedure yields a cluster `wrt`
                `Eps` and `MinPts`
                If `p` is border point, no points are density-reachable from `p` and DBSCAN vistsdef __
                the next point of the database
            """
            self._density_based_algorithm = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean',
                metric_params=None,
                algorithm='auto',
                leaf_size=30,
                p=None,
                n_jobs=1)

            # Display density based scan for debug
            # say.debug('density based debug: {}'.format(self._density_based_algorithm))
            self._cluster = lambda matrix: self._density_based_algorithm.fit_predict(1 - matrix)
        else:
            # Discard error message
            # say.error('required cluster algorithm(Density based)')
            raise TextRankException("required cluster algorithm(Density based)")
        ######################################################
        #               Property(instances)                  #
        ######################################################
        self.non_blow = el.get_word['non_below']
        self.non_above = el.get_word['non_above']
        self.max_dic_length = el.get_hyper_parameter['max_dic_length']
        self.min_cluster_size = el.get_hyper_parameter['min_cluster']
        self.matrix_convex = el.get_hyper_parameter['matrix_convex']
        self.compacted = el.get_hyper_parameter['compacted']
        self.sentence_workspace = KoreanSentenceWorker(korean_corpus_tags,
                                                       korean_corpus_splitter,
                                                       token_min,
                                                       korean_stop_words,
                                                       **kwargs)

    def sentence_processing(self, text):
        import gensim.models
        # Todo text tuple or Dic check(sentences processing)

        self.sentences = self.sentence_workspace.text_to_sentences(text)  # Factory
        self.num_sentences = len(self.sentences)
        self.corpus = KoreanSentenceCorpus(self.sentences, self.non_blow,
                                           self.non_above, self.max_dic_length)

        # Term frequency - Inverse document frequency
        # weight_{i,j} = frequency_{i,j} * log_2 \\frac{D}{document\_freq_{i}}
        self.model = gensim.models.TfidfModel(
            corpus=self.corpus.bows,
            id2word=self.corpus.dictionary,
            normalize=True,
            smartirs=None
        )
        self.tfidfs = self.model[self.corpus.bows]
        ####################################################################################
        #                                  Class Method(Private)                           #
        ####################################################################################
        self.__tf_idf()

        # NOTE
        # Description: Validation Exception input array a minimum of `one` is required
        # Todo: Modify function inter face
        self.__sentences_matrix()
        if len(self.matrix) != 0:
            self.__clustering()
            # if self.compacted: self.__compacted()
            self.__compacted()
            self.graphs = []

            ######################################################
            #                 Algorithm(Page Rank)               #
            ######################################################
            for _ in range(self.num_clusters):
                graph = self.sentences_to_graph(self.clusters[_])
                """ Page Rank Algorithm Powered by Google
                """
                page_rank_algorithm = networkx.pagerank(G=graph, alpha=0.85, personalization=None, tol=1.0e-6,
                                                        max_iter=150, weight='weight', dangling=None)
                self.clusters[_] = sorted(page_rank_algorithm, key=page_rank_algorithm.get, reverse=True)
                """ Graph
                    - A graph is a structure amounting to a set of object in which
                    same some paris of the objects are in som sense `related`
                """
                self.graphs.append(graph)

            return True
        else:
            """ Input Shape Exception
            Description:
            - Clustering through `density based scan` algorithm, we summary metrics isn't convex function and can't cluster sentences

            Todo:
            - cluster algorithm modify or define of `API` interface policy
            """
            return False

    def __cluster_observation(self, k=None):
        global SENTENCES_MAX_LENGTH
        k = max(SENTENCES_MAX_LENGTH, self.num_clusters)
        if operator.gt(k, 1):
            # modify
            # if the number of sentence cluster is less than the sequence length
            # the number of cluster is return without multiplying by `sequences_max_length`
            k = int(self.num_clusters) if int(self.num_sentences) < SENTENCES_MAX_LENGTH else int(self.num_clusters) * k
        else:
            pass
            # Todo exception
        self.k = k

    def __summary_splitter(self):
        summaries = sorted(self.summaries, key=lambda sentence: sentence.index)
        return [(sentence.text, sentence.tokens, sentence.counter) for sentence in summaries]

    def __similarity_jaccard_algorithm(self, s1, s2):
        if s1 == s2:
            return 1
        s1_vote = s1.counter
        s2_vote = s2.counter

        A = sum((s1_vote & s2_vote).values())
        B = sum((s1_vote | s2_vote).values())
        indexes_coefficient = A / B if B else 0

        return indexes_coefficient

    def __exponential_decay(self, s1, s2):
        #  Complete compute distance (sentence, target sentence)
        distance = abs(s1.index - s2.index)
        window_maximum = max(self.decay_window - distance, 0) / self.decay_window

        return math.pow(window_maximum, self.decay_alpha)

    def __tf_idf(self):
        #  Complete
        for _ in range(self.num_sentences):
            bag_of_words = self.corpus.bows[_]
            self.sentences[_].bag_of_words = bag_of_words
            self.sentences[_].tf_idf = self.model[bag_of_words]

    def __sentences_matrix(self):
        #  Complete
        self.matrix = np.zeros((self.num_sentences, self.num_sentences))
        for first_sentence in self.sentences:
            for second_sentence in self.sentences:
                current = first_sentence.index
                next = second_sentence.index
                self.matrix[current, next] = self.multi_sets_sim(first_sentence, second_sentence)
        if self.matrix_convex:
            for _ in range(self.num_sentences):
                self.matrix[_, _] = 0
                self.matrix[_, _] = max(self.matrix[_])
        say.info('\n Build Sentence Matrix {}'.format(self.matrix if len(self.matrix) != 0 else 'required sentence'))
        # ------
        # TODO: Add debug information [level: 1]
        # say.debug(self.matrix if operator.ge(len(self.matrix), 0) else 'required sentence')

    def sentences_to_graph(self, sentences):
        """ Todo Sentence to Page Rank algorithm
        """
        graph = networkx.Graph()
        """
        self, nodes_for_adding, **attr
        """
        graph.add_nodes_from(sentences)
        for first_sentence in sentences:
            for second_sentence in sentences:
                weight = self.matrix[first_sentence.index, second_sentence.index]
                if weight:
                    """
                    :param u_of_edge, v_of_edge:
                        - Nodes must be hashable (and, not None)
                    """
                    graph.add_edge(u_of_edge=first_sentence, v_of_edge=second_sentence, weight=weight)
        return graph

    def __clustered(self):
        # Todo refactor cluster class
        self.clusters = [cluster for cluster in self.clusters if operator.__ge__(len(cluster), self.min_cluster_size)]
        self.num_clusters = len(self.clusters)
        self.clusters = sorted(self.clusters, key=lambda cluster: len(cluster), reverse=True)

    def __clustering(self):
        # Todo refactor cluster class
        cluster_matrix = self._cluster(self.matrix)  # Density Based Spatial Clustering of Application
        bucket = dict()
        for _ in range(cluster_matrix.__len__()):
            key = str(cluster_matrix[_])
            if key not in bucket:
                bucket[key] = []
            bucket[key].append(self.sentences[_])
        self.clusters = bucket.values()
        self.__clustered()

    def __compacted(self):
        clusters = []
        for cluster in self.clusters:  # Todo Cluster unable to get repair for `list`
            compact_cluster = []
            cluster_size = len(cluster)
            for _ in range(cluster_size):
                #  pandas.DataFrame.duplicated([cluster[i]], keep=False)
                cluster[_].duplicated = False

            for _ in range(cluster_size):
                if cluster[_].duplicated:
                    continue
                for j in range(1 + _, cluster_size):
                    if cluster[j].duplicated:
                        continue
                    sentences_degree = self.uniform_similarity(cluster[_], cluster[j])
                    if operator.__gt__(sentences_degree, 0.85):  # Threshold A > B
                        #  pandas.DataFrame.duplicated([cluster[i]], keep=False)
                        cluster[j].duplicated = True
                compact_cluster.append(cluster[_])
            clusters.append(compact_cluster)
        self.clusters = clusters
        self.__clustered()

    def __sum_recursive(self, clusters_investigator, ends):
        # Todo modify function interface
        self.summaries = []
        for _ in range(self.num_clusters):
            # Todo Exception
            self.summaries.append(self.clusters[_][0])
            clusters_investigator[_] += 1
            if operator.__eq__(len(self.summaries), self.k):
                return self.__summary_splitter()

        while True:
            # Todo Exception
            try:
                cluster_branch = np.array([clusters_investigator + 1, ends], dtype=np.float64).min(axis=0) / ends
                branch_arg_min = int(cluster_branch.argmin())
                cluster_investigator = int(clusters_investigator[branch_arg_min])
                self.summaries.append(self.clusters[branch_arg_min][cluster_investigator])

                clusters_investigator[branch_arg_min] += 1
                if len(self.summaries) == self.k:
                    return self.__summary_splitter()
            except:
                pass

    def summary_works(self):
        self.summaries = list()
        self.__cluster_observation()
        if operator.eq(self.k, 0):
            return []
        else:
            ends = np.array([len(cluster) for cluster in self.clusters], dtype=np.int64)
            clusters_investigator = np.zeros(ends.shape, dtype=np.float64)
            for _ in range(self.num_clusters):
                # Todo Exception
                self.summaries.append(self.clusters[_][0])
                clusters_investigator[_] += 1
                if operator.__eq__(len(self.summaries), self.k):
                    return self.__summary_splitter()

            while True:
                # Todo Exception
                try:
                    cluster_branch = np.array([clusters_investigator + 1, ends], dtype=np.float64).min(axis=0) / ends
                    branch_arg_min = int(cluster_branch.argmin())
                    cluster_investigator = int(clusters_investigator[branch_arg_min])
                    self.summaries.append(self.clusters[branch_arg_min][cluster_investigator])

                    clusters_investigator[branch_arg_min] += 1
                    if len(self.summaries) == self.k:
                        return self.__summary_splitter()
                except:
                    # Todo Exception with raise value
                    say.info('Cluster branch tries the sequence if it is not clustered or empty')