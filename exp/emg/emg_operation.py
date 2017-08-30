# -*- coding: utf-8 -*-
import time
import os
import json

import numpy as np
import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS
INTERVAL_TIME = 8


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class Model(object):
    """The RNN model."""

    def __init__(self,  config):
        self.batch_size = config['batch_size']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.input_channel = config['input_channel']
        self.model_structure = config['model_structure']

        self._input_data = tf.placeholder(data_type(), [self.batch_size, self.input_channel, self.input_size])
        self._targets = tf.placeholder(tf.int16, [self.batch_size, self.output_size])
        data = tf.reshape(self._input_data, [self.batch_size, self.input_channel, self.input_size, 1])
        for i, layer in enumerate(self.model_structure):
            data = self.add_conv_layer(
                No=i,
                input=data,
                filter_size=layer["filter_size"],
                out_channels=layer["out_channels"],
            )
            data = self.add_pool_layer(0, data, layer["pool_size"], [1, 1, 1, 1])

        data = tf.reshape(data, [self.batch_size, -1])

        softmax_w = tf.get_variable(
            "softmax_w", [data.shape[1], self.output_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.output_size], dtype=data_type())
        self.logits = logits = tf.matmul(data, softmax_w) + softmax_b
        # print(self.logits.shape)
        self._cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._targets, logits=logits))
        self._predict_op = tf.argmax(logits, 1)

    def add_conv_layer(self, No, input, filter_size, out_channels):
        with tf.variable_scope("conv_layer_%d" % No):
            W = tf.get_variable('filter', [filter_size[0], filter_size[1], input.shape[3], out_channels])
            b = tf.get_variable('bias', [out_channels])
            conv = tf.nn.conv2d(
                input,
                W,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='conv'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        return h

    def add_pool_layer(self, No, input, pool_size, strides):
        for i in range(2):
            if pool_size[i] == -1:
                pool_size[i] = input.shape[1 + i]
        with tf.variable_scope("pool_layer_%d" % No):
            pooled = tf.nn.max_pool(
                input,
                ksize=[1, pool_size[0], pool_size[1], 1],
                padding='VALID',
                strides=strides
            )
        return pooled

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    # @property
    # def initial_state(self):
    #     return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def predict_op(self):
        return self._predict_op


class Emg_operation(object):
    def __init__(self, model_dir):
        config_dir = os.path.join(model_dir, "model_config.json")
        self.config = json.load(open(config_dir))
        self.config["batch_size"] = 1
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
        self.data_index = 0
        self.sample_index = 0
        self.data = np.zeros([self.config["input_channel"], self.config["input_size"]])

    def predict(self):
        fetches = [self.model.predict_op, self.model.logits]
        feed_dict = {}
        data =  np.concatenate([self.data[:, self.data_index:], self.data[:, :self.data_index]], axis=1)
        feed_dict[self.model.input_data] = np.reshape(data, [1, self.data.shape[0], self.data.shape[1]])
        predict_val, logits = self.session.run(fetches, feed_dict)
        # print(predict_val, logits)
        return predict_val[0], logits


    def feed_data(self, data):
        self.data[:, self.data_index] = np.array(data)
        self.data_index += 1
        self.sample_index += 1
        if self.data_index >= self.config["input_size"]:
            self.data_index = 0
        if self.sample_index == INTERVAL_TIME:
            ret, logits = self.predict()
            self.sample_index = 0
            return ret, logits
        else:
            return -1, []