# -*- encoding:utf-8 -*-
from heapq import nlargest
from itertools import product
from gensim.models import Word2Vec
import numpy as np
import os
import codecs
from itertools import count
import jieba
import math
import jieba.posseg as pseg


class FastTextRank4Sentence(object):
    def __init__(self, use_stopword=False, stop_words_file=None, use_w2v=False, dict_path=None, max_iter=100,
                 tol=0.0001):
        """
        :param use_stopword: 是否使用停用词
        :param stop_words_file: 停用词文件路径
        :param use_w2v: 是否使用词向量计算句子相似性
        :param dict_path: 词向量字典文件路径
        :param max_iter: 最大迭代伦茨
        :param tol: 最大容忍误差
        """
        if use_w2v == False and dict_path != None:
            raise RuntimeError("再使用词向量之前必须令参数use_w2v=True")
        self.__use_stopword = use_stopword
        self.__use_w2v = use_w2v
        self.__dict_path = dict_path
        self.__max_iter = max_iter
        self.__tol = tol
        if self.__use_w2v:
            self.__word2vec = Word2Vec.load(self.__dict_path)
        self.__stop_words = set()
        self.__stop_words_file = self.get_default_stop_words_file()
        if type(stop_words_file) is str:
            self.__stop_words_file = stop_words_file
        if use_stopword:
            for word in codecs.open(self.__stop_words_file, 'r', 'utf-8', 'ignore'):
                self.__stop_words.add(word.strip())
        np.seterr(all='warn')  # Print a RuntimeWarning for all types of floating-point errors

    def get_default_stop_words_file(self):
        d = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(d, 'stopwords.txt')

    # 可以改进为删除停用词，词性不需要的词
    def filter_dictword(self, sents):
        """
        删除词向量字典里不存的词
        :param sents:
        :return:
        """
        _sents = []
        dele = set()
        for sentence in sents:
            for word in sentence:
                if word not in self.__word2vec:
                    dele.add(word)
            if sentence:
                _sents.append([word for word in sentence if word not in dele])
        return _sents

    def summarize(self, text, n):
        text = text.replace('\n', '')
        text = text.replace('\r', '')
        text = as_text(text)  # 处理编码问题
        tokens = cut_sentences(text)
        # sentences用于记录文章最原本的句子，sents用于各种计算操作
        sentences, sents = cut_filter_words(tokens, self.__stop_words, self.__use_stopword)
        if self.__use_w2v:
            sents = self.filter_dictword(sents)
        graph = self.create_graph_sentence(sents, self.__use_w2v)
        scores = weight_map_rank(graph, self.__max_iter, self.__tol)
        sent_selected = nlargest(n, zip(scores, count()))
        sent_index = []
        for i in range(n):
            sent_index.append(sent_selected[i][1])  # 添加入关键词在原来文章中的下标
        return [sentences[i] for i in sent_index]

    def create_graph_sentence(self, word_sent, use_w2v):
        """
        传入句子链表  返回句子之间相似度的图
        :param word_sent:
        :return:
        """
        num = len(word_sent)
        board = [[0.0 for _ in range(num)] for _ in range(num)]

        for i, j in product(range(num), repeat=2):
            if i != j:
                if use_w2v:
                    board[i][j] = self.compute_similarity_by_avg(word_sent[i], word_sent[j])
                else:
                    board[i][j] = two_sentences_similarity(word_sent[i], word_sent[j])
        return board

    def compute_similarity_by_avg(self, sents_1, sents_2):
        '''
        对两个句子求平均词向量
        :param sents_1:
        :param sents_2:
        :return:
        '''
        if len(sents_1) == 0 or len(sents_2) == 0:
            return 0.0
        # 把一个句子中的所有词向量相加
        vec1 = self.__word2vec[sents_1[0]]
        for word1 in sents_1[1:]:
            vec1 = vec1 + self.__word2vec[word1]

        vec2 = self.__word2vec[sents_2[0]]
        for word2 in sents_2[1:]:
            vec2 = vec2 + self.__word2vec[word2]

        similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
        return similarity


sentence_delimiters = frozenset(u'。！？……')
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
text_type = str
string_types = (str,)
xrange = range


def as_text(v):  # 生成unicode字符串
    if v is None:
        return None
    elif isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    elif isinstance(v, str):
        return v
    else:
        raise ValueError('Unknown type %r' % type(v))


def is_text(v):
    return isinstance(v, text_type)


def cut_sentences(sentence):
    tmp = []
    for ch in sentence:  # 遍历字符串中的每一个字
        tmp.append(ch)
        if sentence_delimiters.__contains__(ch):
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)


def cut_filter_words(cutted_sentences, stopwords, use_stopwords=False):
    sentences = []
    sents = []
    for sent in cutted_sentences:
        sentences.append(sent)
        if use_stopwords:
            sents.append([word for word in jieba.cut(sent) if word and word not in stopwords])  # 把句子分成词语
        else:
            sents.append([word for word in jieba.cut(sent) if word])
    return sentences, sents


def psegcut_filter_words(cutted_sentences, stopwords, use_stopwords=True, use_speech_tags_filter=True):
    sents = []
    sentences = []
    for sent in cutted_sentences:
        sentences.append(sent)
        jieba_result = pseg.cut(sent)
        if use_speech_tags_filter == True:
            jieba_result = [w for w in jieba_result if w.flag in allow_speech_tags]
        else:
            jieba_result = [w for w in jieba_result]
        word_list = [w.word.strip() for w in jieba_result if w.flag != 'x']
        word_list = [word for word in word_list if len(word) > 0]
        if use_stopwords:
            word_list = [word.strip() for word in word_list if word.strip() not in stopwords]
        sents.append(word_list)
    return sentences, sents


def weight_map_rank(weight_graph, max_iter, tol):
    '''
    输入相似度的图（矩阵)
    返回各个句子的分数
    :param weight_graph:
    :return:
    '''
    # 初始分数设置为0.5
    # 初始化每个句子的分子和老分数
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    denominator = caculate_degree(weight_graph)

    # 开始迭代
    count = 0
    while different(scores, old_scores, tol):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        # 计算每个句子的分数
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, denominator, i)
        count += 1
        if count > max_iter:
            break
    return scores


def caculate_degree(weight_graph):
    length = len(weight_graph)
    denominator = [0.0 for _ in range(len(weight_graph))]
    for j in range(length):
        for k in range(length):
            denominator[j] += weight_graph[j][k]
        if denominator[j] == 0:
            denominator[j] = 1.0
    return denominator


def calculate_score(weight_graph, denominator, i):  # i表示第i个句子
    """
    计算句子在图中的分数
    :param weight_graph:
    :param scores:
    :param i:
    :return:
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        fraction = 0.0
        # 计算分子
        # [j,i]是指句子j指向句子i
        fraction = weight_graph[j][i] * 1.0
        # 除以j的出度
        added_score += fraction / denominator[j]
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def different(scores, old_scores, tol=0.0001):
    '''
    判断前后分数有无变化
    :param scores:
    :param old_scores:
    :return:
    '''
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= tol:  # 原始是0.0001
            flag = True
            break
    return flag


def cosine_similarity(vec1, vec2):
    '''
    计算两个向量之间的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


def combine(word_list, window=2):
    """构造在window下的单词组合，用来构造单词之间的边。

    Keyword arguments:
    word_list  --  list of str, 由单词组成的列表。
    windows    --  int, 窗口大小。
    """
    if window < 2: window = 2
    for x in xrange(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r


def two_sentences_similarity(sents_1, sents_2):
    '''
    计算两个句子的相似性
    :param sents_1:
    :param sents_2:
    :return:
    '''
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    if counter == 0:
        return 0
    return counter / (math.log(len(sents_1) + len(sents_2)))
