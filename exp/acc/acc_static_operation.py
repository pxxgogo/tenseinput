# -*- coding: utf-8 -*-
import time
import os
import json

import numpy as np
import tensorflow as tf
from model import Model

INTERVAL_TIME = 300
WINDOW_SIZE = 400


def fft(window_data):
    x_freq = abs(np.fft.rfft(np.array(window_data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(window_data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(window_data[2], dtype=np.float32)))
    freqs = [x_freq, y_freq, z_freq]
    return np.array(freqs)

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class Acce_operation(object):
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
        self.data = np.zeros([self.config["input_channel"], WINDOW_SIZE])

    def predict(self):
        fetches = [self.model.predict_op, self.model.logits]
        feed_dict = {}
        data = np.concatenate([self.data[:, self.data_index:], self.data[:, :self.data_index]], axis=1)
        freqs = fft(data)
        feed_dict[self.model.input_data] = np.reshape(freqs, [1, freqs.shape[0], freqs.shape[1]])
        predict_val, logits = self.session.run(fetches, feed_dict)
        print(predict_val, logits, logits[0][0] - logits[0][1])
        return predict_val[0], logits


    def feed_data(self, data):
        self.data[:, self.data_index] = np.array(data)
        self.data_index += 1
        self.sample_index += 1
        if self.data_index >= WINDOW_SIZE:
            self.data_index = 0
        if self.sample_index == INTERVAL_TIME:
            ret, logits = self.predict()
            self.sample_index = 0
            return ret, logits
        else:
            return -1, []