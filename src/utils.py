# -*- coding: utf-8 -*-

"""
Created on 2020-07-21 14:25
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

import codecs

from config import Config
from keras_bert import Tokenizer
import numpy as np
from keras.utils import to_categorical



class data_loader(object):
    """
    将数据自文件中加载出来，若非加载预设档案时这里需要更改
    """
    def __init__(self):
        self.config = Config()
        self.cates_dict = self.get_category_id()

    def get_category_id(self):
        """
        返回分类种类的索引
        :return: 返回分类种类的字典
        """
        # categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
        categories = ['电视剧', '娱乐', '动漫', '电影', '综艺', '教育', '网络电影', '纪录片', '财经', '片花', '音乐', '军事', '游戏', '资讯']
        cates_dict = dict(zip(categories, range(len(categories))))
        return cates_dict

    def get_clf_idx(self, key):
        return self.cates_dict.get(key)

    def make_data(self, data_path):
        with open(data_path, "r", encoding='utf-8') as f:
            lines = [line.replace("\n", "") for line in f.readlines()]

        data = []
        for idx in range(len(lines)):
            tmp = lines[idx].split('\t')
            label = tmp[0]
            text = tmp[1]
            data.append([self.get_clf_idx(label), text])
        return data


class OurTokenizer(Tokenizer):
    """
    重写Tokenizer中的函数
    """
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class data_generator:
    """
    批量数据生成，减少内存消耗的写法
    """
    def __init__(self, data, max_length, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.maxlen = max_length # 不得大于512
        self.token_dict = {}
        self.load_token_dict()
        self.tokenizer = OurTokenizer(self.token_dict)
        # self.config = Config()

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for idx in idxs:
                current_sample = self.data[idx]
                y = current_sample[0]
                text = current_sample[1][:self.maxlen]
                x1, x2 = self.tokenizer.encode(first=text)

                X1.append(x1)
                X2.append(x2)
                Y.append(y)

                if len(X1) == self.batch_size or idx == idxs[-1]:
                    X1 = self.seq_padding(X1)
                    X2 = self.seq_padding(X2)

                    Y = self.seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

    def seq_padding(self, X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)

        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])

    def load_token_dict(self):
        dict_path = '../bert/chinese_L-12_H-768_A-12/vocab.txt'
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(token)


def build_data_generator_input(file_path):
    loader = data_loader()
    data = loader.make_data(file_path)
    num_class = len(loader.get_category_id())
    result = []
    for data_row in data:
        label = data_row[0]
        content = data_row[1]
        # 直接声明label维度大小，避免因样本不平衡导致输出维度与验证维度不吻合
        result.append([to_categorical(label, num_classes=num_class), content])
    return np.array(result)

if __name__ == '__main__':
    test = data_loader()
    print(test.make_data('../data/cgc/train.txt')[0])