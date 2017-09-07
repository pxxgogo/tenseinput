# -*- coding: utf-8 -*-


import time
import os

import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
import json
from model import Model

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
        self.input_channel = self.config["input_channel"][0]
        self.sequence_size = self.config.get("sequence_size", [1, 1])[0]

    def predict(self, data):
        fetches = [self.model.predict_op, self.model.logits]
        feed_dict = {}
        feed_dict[self.model.input_data] = data
        predict_val, logits = self.session.run(fetches, feed_dict)
        print(predict_val, logits, logits[0][0] - logits[0][1])
        return predict_val[0], logits


