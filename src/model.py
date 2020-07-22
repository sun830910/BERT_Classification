# -*- coding: utf-8 -*-

"""
Created on 2020-07-21 14:05
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com
"""

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

from config import Config
from utils import data_loader, build_data_generator_input, data_generator




class multi_clf_model(object):
    def __init__(self):
        self.config = Config()
        self.config_path = self.config.get("BERT_path", "config_path")
        self.checkpoint_path = self.config.get("BERT_path", "checkpoint_path")

        self.trainingSet_path = self.config.get("data_path", "trainingSet_path")
        self.testingSet_path = self.config.get("data_path", "testingSet_path")
        self.val_path = self.config.get("data_path", "valSet_path")

        self.model_save_path = self.config.get("result", "model_save_path")

    def model(self, num_class, trainable):
        config_path = self.config_path
        checkpoint_path = self.checkpoint_path

        bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

        for l in bert_model.layers:
            l.trainable = trainable

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)
        p = Dense(num_class, activation='softmax')(x)

        model = Model([x1_in, x2_in], p)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(1e-5),  # 用足够小的学习率
                      metrics=['accuracy'])

        model.summary()
        return model

    def train(self, num_class, epochs, trainable):

        # 训练数据、测试数据和标签转化为模型输入格式
        training_data_list = build_data_generator_input(self.trainingSet_path)
        val_data_list = build_data_generator_input(self.val_path)
        # 测试集写好了
        # testing_data_list = build_data_generator_input(self.testingSet_path)

        max_length = self.config.get("training_rule", "max_length")

        train_D = data_generator(training_data_list, max_length=max_length, shuffle=True)
        valid_D = data_generator(val_data_list, max_length=max_length, shuffle=True)
        # test_D = data_generator(testing_data_list, shuffle=False)

        model = self.model(num_class, trainable)
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=epochs,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D)
        )
        model.save()