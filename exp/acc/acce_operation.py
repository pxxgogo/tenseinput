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
SAMPLE_NUM = 2
DEVICES = "/cpu:0"


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class Model(object):
    """The RNN model."""

    def __init__(self, config):
        self.batch_size = 1
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.input_channel = config['input_channel']
        self.model_structure = config['model_structure']

        self._input_data = tf.placeholder(data_type(), [self.batch_size, self.input_channel, self.input_size])
        self._targets = tf.placeholder(tf.int16, [self.batch_size, self.output_size])
        data = self._input_data
        layer_No = 0
        for i, layer in enumerate(self.model_structure):
            net_type = layer["net_type"]
            repeated_times = layer.get("repeated_times", 1)
            while repeated_times > 0:
                if net_type == "LSTM":
                    data = self.add_lstm_layer(
                        No=layer_No,
                        input=data,
                        hidden_size=layer["hidden_size"]
                    )
                elif net_type == "RESNET":
                    data = self.add_renet_layer(data, self._input_data)
                elif net_type == "CNN":
                    # print(data.shape)
                    # print(repeated_times)
                    if len(data.shape) == 3:
                        data = tf.reshape(data, [self.batch_size, self.input_channel, -1, 1])

                    data = self.add_conv_layer(
                        No=layer_No,
                        input=data,
                        filter_size=layer["filter_size"],
                        out_channels=layer["out_channels"],
                        filter_type=layer["filter_type"]
                    )
                    data = self.add_pool_layer(layer_No, data, layer["pool_size"], [1, 1, 1, 1],
                                               pool_type=layer["pool_type"])
                repeated_times -= 1
                layer_No += 1
        #
        print(data.shape)
        # exit()

        data = tf.reshape(data, [self.batch_size, -1])

        softmax_w = tf.get_variable(
            "softmax_w", [data.shape[1], self.output_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.output_size], dtype=data_type())
        self.logits = logits = tf.matmul(data, softmax_w) + softmax_b
        # print(self.logits.shape)
        self._cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._targets, logits=logits))
        self._predict_op = tf.argmax(logits, 1)

    def add_lstm_layer(self, No, input, hidden_size):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                 reuse=tf.get_variable_scope().reuse)
        with tf.variable_scope("lstm_layer_%d" % No):
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                dtype=data_type(),
                inputs=input)
        return tf.convert_to_tensor(outputs)

    def add_conv_layer(self, No, input, filter_size, out_channels, filter_type):
        with tf.variable_scope("conv_layer_%d" % No):
            W = tf.get_variable('filter', [filter_size[0], filter_size[1], input.shape[3], out_channels])
            b = tf.get_variable('bias', [out_channels])
            conv = tf.nn.conv2d(
                input,
                W,
                strides=[1, 1, 1, 1],
                padding=filter_type,
                name='conv'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        return h

    def add_pool_layer(self, No, input, pool_size, strides, pool_type):
        for i in range(2):
            if pool_size[i] == -1:
                pool_size[i] = input.shape[1 + i]
        with tf.variable_scope("pool_layer_%d" % No):
            pooled = tf.nn.max_pool(
                input,
                ksize=[1, pool_size[0], pool_size[1], 1],
                padding=pool_type,
                strides=strides,
                name='pool'
            )
        return pooled

    def add_renet_layer(self, data, input):
        return data + input

    @property
    def input_data(self):
        return self._input_data

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
