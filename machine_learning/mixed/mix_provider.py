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
        'max_epoch': 150,
        'keep_prob': 1.0,
        'lr_decay': 0.1,
        'batch_size': 16,
        'input_size': INPUT_SIZE,
        'output_size': OUTPUT_SIZE
    }
    CORPUS_CONFIG_NAME = "corpus_config.json"
    FILENAMES = ["training_data.npy", "test_data.npy"]

    def __init__(self, config_dir):
        self.data_dir = ''
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
        self.model_config["ignored_gestures"] = self.ignored_gestures = config.get("ignored_gestures", [])
        self.model_config["ignored_users"] = self.ignored_users = config.get("ignored_users", [])
        self.model_config["acc_model_structure"] = config["acc_model_structure"]
        self.model_config["emg_model_structure"] = config["emg_model_structure"]
        self.model_config["learning_rate"] = config["learning_rate"]
        self.model_config["model_type"] = config["model_type"]
        self.model_config["acc_regularized_lambda"] = config.get("acc_regularized_lambda", 0.000001)
        self.model_config["emg_regularized_lambda"] = config.get("emg_regularized_lambda", 0.000001)
        self.model_config["regularized_flag"] = config.get("regularized_flag", True)
        self.model_config["data_dir"] = self.data_dir
        self.output_type = config["output_type"]
        print("finish parsing config")

    def get_trainable_data(self, data):
        if self.output_type == 0:
            return [np.array(list(data[:, 0])), np.array(list(data[:, 1]))]
        else:
            new_data = []
            for sub_data in data:
                if sub_data[2] in self.ignored_gestures:
                    continue
                if sub_data[3] in self.ignored_users:
                    continue
                new_data.append(sub_data)
            data = np.array(new_data)
            x = np.array(list(data[:, 0]))
            raw_y = data[:, 1]
            gesture_types = data[:, 2]
            users = data[:, 3]
            sample_ids = data[:, 4]
            y = []
            for sub_raw_y in raw_y:
                sub_y = [0 for i in range(self.output_size)]
                sub_y[sub_raw_y] = 1
                y.append(sub_y)
            return x, np.array(y), gesture_types, users, sample_ids

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
        print("epoch_size", epoch_size)
        if self.status == 'train':
            for i in range(epoch_size):
                x = self.training_data[0][i * self.batch_size: (i + 1) * self.batch_size]
                y = self.training_data[1][i * self.batch_size: (i + 1) * self.batch_size]
                gesture_types = self.training_data[2][i * self.batch_size: (i + 1) * self.batch_size]
                users = self.training_data[3][i * self.batch_size: (i + 1) * self.batch_size]
                sample_ids = self.training_data[4][i * self.batch_size: (i + 1) * self.batch_size]
                yield (x, y, gesture_types, users, sample_ids)
        else:
            for i in range(epoch_size):
                x = self.test_data[0][i: (i + 1)]
                y = self.test_data[1][i: (i + 1)]
                gesture_types = self.test_data[2][i: (i + 1)]
                users = self.test_data[3][i: (i + 1)]
                sample_ids = self.test_data[4][i: (i + 1)]
                yield (x, y, gesture_types, users, sample_ids)


if __name__ == "__main__":
    '''
    Debug
    '''
    provide = Provider('./config.json')
    provide.status = 'train'
    for x, y in provide():
        print("input", x.shape)
        print("output", y.shape)
        input("Next")
