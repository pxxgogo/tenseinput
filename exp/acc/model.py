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
                elif net_type == "DENSE":
                    keep_prob = 1.0
                    data = self.add_dense_layer(
                        No=layer_No,
                        input=data,
                        output_size=layer["output_size"],
                        keep_prob=keep_prob
                    )
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

    def get_length(self, input):
        ret = 1
        for i in range(1, len(input.shape)):
            ret *= int(input.shape[i])
        return ret

    def add_dense_layer(self, No, input, output_size, keep_prob):
        with tf.variable_scope("dense_layer_%d" % No):
            input_length = self.get_length(input)
            W = tf.get_variable('dense', [input_length, output_size])
            b = tf.get_variable('bias', [output_size])
            data = tf.reshape(input, [-1, int(input_length)])
            data = tf.nn.relu(tf.matmul(data, W) + b)
            if keep_prob < 1.0:
                data = tf.nn.dropout(data, keep_prob)
        return data

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
