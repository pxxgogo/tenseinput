# -*- coding: utf-8 -*-


import time
import os

import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
import json
from model import Model

SAMPLE_NUM = 2
class Acce_operation(object):
    def __init__(self, model_dir):
        config_dir = os.path.join(model_dir, "model_config.json")
        self.config = json.load(open(config_dir))
        session_config = tf.ConfigProto(log_device_placement=False)
        session_config.gpu_options.allow_growth = False
        self.session = tf.Session(config=session_config)
        with tf.variable_scope("model", reuse=None):
            self.model = Model(config=self.config)
        self.session.run(tf.global_variables_initializer())
        new_saver = tf.train.Saver()
        new_saver.restore(self.session, tf.train.latest_checkpoint(
            model_dir))
        for v in tf.global_variables():
            print(v.name)
        self.input_channel = self.config["input_channel"]
        # print(self.input_channel)
        self.data = np.zeros([1, self.input_channel, self.config["input_size"]])
        self.index = 0
        self.sample_index = 0
        self.debug_index_list = np.zeros([1, self.input_channel])
        self.debug_index = 0

    def operate_data(self, data):
        op_data = data[:, :self.config["input_size"] // 3]
        op_data = np.reshape(op_data, (-1))
        # print(op_data.shape)
        return op_data

    def clean(self):
        self.data = np.zeros([1, self.input_channel, self.config["input_size"]])
        self.index = 0
        self.sample_index = 0
        self.debug_index_list = np.zeros([1, self.input_channel])
        self.debug_index = 0

    def clean_partly(self):
        self.debug_index_list = np.zeros([1, self.input_channel])
        self.debug_index = 0

    def feed_data(self, data):
        data = self.operate_data(data)
        self.data[0][self.index] = data[:]
        self.debug_index_list[0][self.index] = self.debug_index
        self.sample_index += 1
        self.index += 1
        self.debug_index += 1
        if self.index == self.input_channel:
            self.index = 0
        if self.sample_index == SAMPLE_NUM:
            self.sample_index = 0
            # print(self.data[0, 0, 0:50])
            window_data = np.concatenate([self.data[:, self.index:], self.data[:, :self.index]], axis=1)
            debug_index_list = np.concatenate(
                [self.debug_index_list[:, self.index:], self.debug_index_list[:, :self.index]], axis=1)
            # print(window_data[0])
            # print(debug_index_list)
            return self.rnn_classify(window_data)
        else:
            return -1, -1

    def rnn_classify(self, data):
        fetches = [self.model.predict_op, self.model.logits]
        feed_dict = {}
        feed_dict[self.model.input_data] = data
        predict_val, logits = self.session.run(fetches, feed_dict)
        return predict_val[0], logits

# if __name__ == "__main__":
#     main()
