# -*- encoding:utf-8 -*-
import re
import os
from fastapi import FastAPI
from FastTextRank4Sentence import FastTextRank4Sentence
app = FastAPI()


class Predictor():
    def __init__(self, max_len=500):
        self.max_len = max_len  # 将整个文本拆分成在max_len左右寻找断句的符号
        self.stop_words_path = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."),
            "stopwords.txt")
        self.mod = FastTextRank4Sentence(use_stopword=True, stop_words_file=self.stop_words_path, use_w2v=False,
                                         tol=0.001)

    def encoder(self, text):
        # 分句
        text = re.sub('([。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)  # 英文省略号
        text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)  # 中文省略号
        text = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        text.split("\n")
        # 按指定 max_len 合并句子长度
        str = ""
        result = []
        for s in text:
            str += s
            if len(str) <= self.max_len:
                continue
            else:
                result.append(str)
                str = ""
        result_2 = ""
        for s in result:
            res_text = self.mod.summarize(s, 1)
            result_2 += res_text[0] + "\b"
        return {"answer": result_2}


@app.get("/{text}")
def read_item(text, max_len: int = 512):
    pre = Predictor(max_len)
    result = pre.encoder(str(text))
    return result