import numpy as np
import json
import sys
from os import path as op
import os
import traceback
import random

INPUT_SIZE = 180
OUTPUT_SIZE = 2
PADDING = 2


class Provider(object):
    model_sample = {
        'init_scale': 0.1,
        'max_grad_norm': 5,
        'max_epoch': 300,
        'keep_prob': 1.0,
        'lr_decay': 0.1,
        'batch_size': 16,
        'input_channel': 1,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE
    }
    CORPUS_CONFIG_NAME = "corpus_config.json"
    FILENAMES = ["training_data.npy", "test_data.npy"]

    def __init__(self, config_dir):
        self.data_dir = ''
        self.legal_model_size = ['s', 'm', 'l', 't']
        self.input_compat = raw_input if sys.version_info[0] < 3 else input
        self.data_dir = ''
        self.model = ''
        self.status = 'IDLE'

        self.batch_size = 1
        self.yield_pos = [0, 0, 0]

        self._parse_config(config_dir)
        self._read_data()

    def _parse_config(self, config_dir):
        with open(config_dir, 'r') as config_handle:
            config = json.load(config_handle)
        self.data_dir = config['data_dir']
        data_config_dir = op.join(self.data_dir, Provider.CORPUS_CONFIG_NAME)
        with open(data_config_dir, 'r') as config_handle:
            self.data_config = json.load(config_handle)
        self.model_config = Provider.model_sample
        self.model_config["batch_size"] = self.batch_size = config['batch_size']
        self.model_config["input_size"] = self.input_size = config["input_size"]
        self.model_config["output_size"] = self.output_size = config["output_size"]
        self.model_config["input_channel"] = self.input_channel = config["input_channel"]
        self.model_config["model_structure"] = config["model_structure"]
        self.model_config["learning_rate"] = config["learning_rate"]
        self.model_config["sliding_flag"] = self.sliding_flag = config.get("sliding_flag", 0)
        self.model_config["regularized_lambda"] = config.get("regularized_lambda", 0.000001)
        self.output_type = config["output_type"]
        print("finish parsing config")

    def get_trainable_data(self, data):
        if self.sliding_flag == 1:
            if self.output_type == 0:
                x = []
                y = []
                for sub_data in data:
                    x.append(sub_data[0][PADDING: PADDING + self.input_channel])
                    y.append(sub_data[1])
                return [np.array(x), np.array(y)]
            else:
                x = np.array(list(data[:, 0]))
                raw_y = data[:, 1]
                y = []
                for sub_raw_y in raw_y:
                    sub_y = [0 for i in range(self.output_size)]
                    sub_y[sub_raw_y] = 1
                    y.append(sub_y)
                return x[:, PADDING: PADDING + self.input_channel], np.array(y)
        else:
            if self.output_type == 0:
                return [np.array(list(data[:, 0])), np.array(list(data[:, 1]))]
            else:
                x = np.array(list(data[:, 0]))
                raw_y = data[:, 1]
                y = []
                for sub_raw_y in raw_y:
                    sub_y = [0 for i in range(self.output_size)]
                    sub_y[sub_raw_y] = 1
                    y.append(sub_y)
                return x, np.array(y)

    def get_config(self):
        return self.model_config

    def _read_data(self):
        self.raw_training_data = np.load(op.join(self.data_dir, self.data_config["training_data"]))
        self.test_data = self.get_trainable_data(np.load(op.join(self.data_dir, self.data_config["test_data"])))

    def get_epoch_size(self):
        if self.status == 'train':
            return (len(self.training_data[0])) // self.batch_size - 1
        elif self.status == 'test':
            return len(self.test_data[0]) - 1
        else:
            return None

    def __call__(self):
        self.status = self.status.strip().lower()
        np.random.shuffle(self.raw_training_data)
        self.training_data = self.get_trainable_data(self.raw_training_data)
        epoch_size = self.get_epoch_size()
        # print("epoch_size", epoch_size)
        if self.status == 'train':
            for i in range(epoch_size):
                x = self.training_data[0][i * self.batch_size: (i + 1) * self.batch_size]
                y = self.training_data[1][i * self.batch_size: (i + 1) * self.batch_size]
                yield (x, y)
        else:
            for i in range(epoch_size):
                x = self.test_data[0][i: (i + 1)]
                y = self.test_data[1][i: (i + 1)]
                yield (x, y)


if __name__ == "__main__":
    '''
    Debug
    '''
    provide = Provider('./acc/config_static_6_channels.json')
    provide.status = 'train'
    for x, y in provide():
        print("input", x.shape)
        print("output", y)
        input("Next")
