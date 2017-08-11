import numpy as np
import json
import sys
from os import path as op
import os
import traceback
import random


class Provider(object):
    model_sample = {
        's': {
            'init_scale': 0.1,
            'learning_rate': 1.0,
            'max_grad_norm': 5,
            'num_layers': 2,
            'hidden_size': 200,
            'max_epoch': 4,
            'max_max_epoch': 13,
            'keep_prob': 1.0,
            'lr_decay': 0.5,
            'batch_size': 16,
        },
        'm': {
            'init_scale': 0.05,
            'learning_rate': 1.0,
            'max_grad_norm': 5,
            'num_layers': 2,
            'hidden_size': 768,
            'max_epoch': 6,
            'max_max_epoch': 39,
            'keep_prob': 0.5,
            'lr_decay': 0.8,
            'batch_size': 128,
        },
        'l': {
            'init_scale': 0.04,
            'learning_rate': 1.0,
            'max_grad_norm': 10,
            'num_layers': 2,
            'hidden_size': 1536,
            'max_epoch': 14,
            'max_max_epoch': 55,
            'keep_prob': 0.35,
            'lr_decay': 1. / 1.15,
            'batch_size': 128,
        },
        't': {
            'init_scale': 0.05,
            'learning_rate': 1.0,
            'max_grad_norm': 5,
            'num_layers': 2,
            'hidden_size': 384,
            'max_epoch': 6,
            'max_max_epoch': 39,
            'keep_prob': 0.5,
            'lr_decay': 0.8,
            'batch_size': 32,
        },
    }

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
        config = json.load(open(config_dir, 'r'))
        self.data_dir = config['data_dir']
        self.model = config['model']
        self.model = self.model[0].lower()
        assert self.model in self.legal_model_size and op.isdir(self.data_dir) and os.access(self.data_dir, os.R_OK)
        self.data_config_dir = op.join(self.data_dir, config['data_config_dir'])
        self.model_config = Provider.model_sample[self.model]
        self.batch_size = self.model_config['batch_size']
        print("finish parsing config")

    def generate_data(self, tag, file_dir):
        raw_data = np.load(file_dir)
        new_data = []
        for data in raw_data:
            op_data = data[:, :, :150]
            op_data = np.reshape(op_data, (op_data.shape[0], -1))
            new_data.append([op_data, tag])
        return new_data

    def get_trainable_data(self, data):
        x = []
        y = []
        for sub_data in data:
            x.append(sub_data[0])
            y.append(sub_data[1])
        return [x, y]

    def _read_data(self):
        self.data_config = json.load(open(self.data_config_dir, 'r'))
        self.data = []
        for data_setting in self.data_config:
            sub_data = self.generate_data(data_setting['tag'], op.join(self.data_dir, data_setting['file_name']))
            self.data.extend(sub_data)
        random.shuffle(self.data)
        training_num = len(self.data) // 10 * 9
        self.training_data = self.get_trainable_data(self.data[:training_num])
        self.test_data = self.get_trainable_data(self.data[training_num:])



    def get_config(self):
        return self.model_config

    def get_epoch_size(self):
        if self.status == 'train':
            return (len(self.training_data[0])) // self.batch_size - 1
        elif self.status == 'test':
            return len(self.test_data[0]) - 1
        else:
            return None



    def __call__(self):
        self.status = self.status.strip().lower()
        epoch_size = self.get_epoch_size()
        print("epoch_size", epoch_size)
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
    provide = Provider('./config.json')
    provide.status = 'train'
    for x, y in provide():
        print("input", x)
        print("output", y)
        input("Next")
