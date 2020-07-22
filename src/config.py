# -*- coding: utf-8 -*-

"""
Created on 2020-07-19 00:20
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

配置模型、路径、与训练相关参数
"""

class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                "vocab_path": "../data/cnews.vocab.txt",
                "trainingSet_path": "../data/cnews.train.txt",
                "valSet_path": "../data/cnews.val.txt",
                "testingSet_path": "../data/cnews.test.txt"
            },

            "BERT_path": {
                "config_path": '../bert/chinese_L-12_H-768_A-12/bert_config.json',
                "checkpoint_path": '../bert/chinese_L-12_H-768_A-12/bert_model.ckpt',
            },

            "training_rule": {
                "max_length": 300 # 输入序列长度，别超过512
            },

            "result": {
                "model_save_path": '../result/bert_keras.h5'
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]