# -*- coding: utf-8 -*-


import time
import os

import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
import json

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
SAMPLE_NUM = 1
DEVICES = "/cpu:0"


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class Model(object):
    """The RNN model."""

    # def __init__(self, is_training, config):
    #     self.batch_size = 1
    #     self.hidden_size = config['hidden_size']
    #     self.input_size = config['input_size']
    #     self.output_size = config['output_size']
    #
    #     self._input_data = tf.placeholder(data_type(), [self.batch_size, self.input_size])
    #
    #     # Slightly better results can be obtained with forget gate biases
    #     # initialized to 1 but the hyperparameters of the model would need to be
    #     # different than reported in the paper.
    #
    #
    #     with tf.device(DEVICES):
    #         lstm_cell_list = []
    #         for i in range(config['num_layers']):
    #             lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True,
    #                                                      reuse=tf.get_variable_scope().reuse)
    #             if is_training and config['keep_prob'] < 1:
    #                 lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config['keep_prob'])
    #             lstm_cell_list.append(lstm_cell)
    #
    #         cell = tf.contrib.rnn.MultiRNNCell(lstm_cell_list, state_is_tuple=True)
    #
    #     self._initial_state = cell.zero_state(self.batch_size, data_type())
    #
    #     inputs = self._input_data
    #
    #     state = self._initial_state
    #     with tf.variable_scope("RNN"):
    #         (cell_output, state) = cell(inputs, state)
    #
    #     output = cell_output
    #     softmax_w = tf.get_variable(
    #         "softmax_w", [self.hidden_size, self.output_size], dtype=data_type())
    #     softmax_b = tf.get_variable("softmax_b", [self.output_size], dtype=data_type())
    #     logits = tf.matmul(output, softmax_w) + softmax_b
    #     self._final_state = state
    #
    #     self._predict_op = tf.argmax(logits, 1)

    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.input_size = config['input_size']
        self.num_steps = config['num_steps']
        self.output_size = config['output_size']
        self.batch_size = 1

        self._input_data = tf.placeholder(data_type(), [self.batch_size, self.num_steps, self.input_size])

        with tf.device(DEVICES):
            lstm_cell_list = []
            for i in range(config['num_layers']):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                         reuse=tf.get_variable_scope().reuse)
                lstm_cell_list.append(lstm_cell)
            cell = tf.contrib.rnn.MultiRNNCell(lstm_cell_list, state_is_tuple=True)
        inputs = self._input_data
        # self._initial_state = cell.zero_state(self.batch_size, data_type())
        with tf.variable_scope("RNN"):
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=data_type(),
                inputs=inputs)
        # print(outputs.shape)
        self.debug_val = outputs
        self.output = output = outputs[:, -1, :]
        print(output.shape)
        softmax_w = tf.get_variable(
            "softmax_w", [self.hidden_size, self.output_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.output_size], dtype=data_type())
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self._predict_op = tf.argmax(self.logits, 1)

    @property
    def input_data(self):
        return self._input_data

    @property
    def initial_state(self):
        return self._initial_state

    # @property
    # def final_state(self):
    #     return self._final_state

    @property
    def predict_op(self):
        return self._predict_op


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
        self.num_steps = self.config["num_steps"]
        # print(self.num_steps)
        self.data = np.zeros([1, self.num_steps, self.config["input_size"]])
        self.index = 0
        self.sample_index = 0
        self.debug_index_list = np.zeros([1, self.num_steps])
        self.debug_index = 0

    def operate_data(self, data):
        op_data = data[:, :self.config["input_size"] // 3]
        op_data = np.reshape(op_data, (-1))
        # print(op_data.shape)
        return op_data

    def clean(self):
        self.data = np.zeros([1, self.num_steps, self.config["input_size"]])
        self.index = 0
        self.sample_index = 0
        self.debug_index_list = np.zeros([1, self.num_steps])
        self.debug_index = 0

    def clean_partly(self):
        self.debug_index_list = np.zeros([1, self.num_steps])
        self.debug_index = 0

    def feed_data(self, data):
        data = self.operate_data(data)
        self.data[0][self.index] = data[:]
        self.debug_index_list[0][self.index] = self.debug_index
        self.sample_index += 1
        self.index += 1
        self.debug_index += 1
        if self.index == self.num_steps:
            self.index = 0
        if self.sample_index == SAMPLE_NUM:
            self.sample_index = 0
            # print(self.data[0, 0, 0:50])
            window_data = np.concatenate([self.data[:, self.index:], self.data[:, :self.index]], axis=1)
            debug_index_list = np.concatenate([self.debug_index_list[:, self.index:], self.debug_index_list[:, :self.index]], axis=1)
            # print(window_data[0])
            # print(debug_index_list)
            return self.rnn_classify(window_data)
        else:
            return -1, -1

    def rnn_classify(self, data):
        fetches = [self.model.predict_op, self.model.logits, self.model.debug_val]
        feed_dict = {}
        feed_dict[self.model.input_data] = data
        predict_val, logits, debug_val = self.session.run(fetches, feed_dict)
        # print(data[0, 0, 0:50])
        # print(debug_val[0, 0, 0:10])
        return predict_val[0], logits

# if __name__ == "__main__":
#     main()
