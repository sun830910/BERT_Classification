# -*- coding: utf-8 -*-

"""
Created on 2020-07-22 17:26
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""
from model import multi_clf_model

if __name__ == '__main__':
    bert_clf = multi_clf_model()
    bert_clf.train(num_class=14, epochs=1, trainable=True)