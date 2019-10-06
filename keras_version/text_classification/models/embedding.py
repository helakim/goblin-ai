from gensim.models import Word2Vec


class W2vGraph(object):
    """
    """

    def __init__(self, model_path, text_list, embedding_size, type, loss, min_count,
                 window_size):
        """
        :param model_path: trained model save path
        :param text_list: input text list
        :param embedding_size: word embedding size
        :param type: 0 for CBOW, 1 for skip-gram
        :param loss: 0 for negative sampling, 1 for hierarchical softmax
        :param min_count:
        :param window_size:
        Description:
            - Word2vec models are used to produce word embeddings
        Reference:
            - Distributed Representations of Words and Phrases and their Compositionality
                - https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

        """
        super(W2vGraph, self).__init__()
        # ------
        # instance variable
        self.model_path = model_path
        self.text_list = text_list
        self.type = type
        self.embedding_size = embedding_size
        self.loss = loss
        self.min_count = min_count
        self.window_size = window_size

    def op_train(self):
        text_list = []
        for i in self.text_list:
            temp = []
            for _i in i:
                temp.append(str(_i))
            text_list.append(temp)
        model = Word2Vec(sentences=text_list, size=self.embedding_size, window=self.window_size,
                         min_count=self.min_count, sg=self.type, hs=self.loss)

        model.save(self.model_path)
