import datetime

import numpy as np
import time
import json
from acce_operation import Acce_operation
import argparse

debug_No = 0
WINDOW_SIZE = 1500
SAMPLE_SIZE = 400
SPLIT_NUM = 7500


class Provider(object):
    def __init__(self, tag, data_dir):
        self.tag = tag
        self.data = json.load(open(data_dir))
        if tag == 0:
            self.data = np.array(self.data[0])
        self.index = 0

    def __call__(self, *args, **kwargs):
        if self.tag == 0:
            epoch = self.data.shape[1] // SPLIT_NUM
            for i in range(epoch):
                data = self.data[:, i * SPLIT_NUM: (i + 1) * SPLIT_NUM]
                yield data
        else:
            for data in self.data:
                yield np.array(data[0])

    def check_ans(self, ret):
        if ret == self.tag:
            return True
        else:
            return False


def operation(data, acce_operation):
    x_freq = abs(np.fft.rfft(np.array(data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(data[2], dtype=np.float32)))
    freqs = np.array([x_freq, y_freq, z_freq])
    ret, debug_val = acce_operation.feed_data(freqs)
    global debug_No
    if not ret == -1:
        print(debug_No, ret, debug_val)
        # print(freqs[:, 0])
        debug_No += 1
    if debug_No == 677:
        exit()


def run(acce_operation, data, window_data, index, sample_num):
    acce_index = 0
    while True:
        for i in range(3):
            window_data[i][index] = data[i, acce_index]
        acce_index += 1
        if acce_index == data.shape[1]:
            break
        index += 1
        sample_num += 1
        if index == WINDOW_SIZE:
            index = 0
        if sample_num == SAMPLE_SIZE:
            sample_num = 0
            ret_data = np.concatenate([window_data[:, index:], window_data[:, :index]], axis=1)
            operation(ret_data, acce_operation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('tag', type=int)
    parser.add_argument('--model_dir', type=str, default="../model/rnn_acce_5")
    args = parser.parse_args()
    model_dir = args.model_dir
    data_dir = args.data_dir
    tag = args.tag
    acce_operation = Acce_operation(model_dir)
    provider = Provider(tag, data_dir)
    window_data = np.array([[0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)],
                            [0 for i in range(WINDOW_SIZE)]])
    index = 0
    sample_num = 0
    for data in provider():
        print(data.shape)
        run(acce_operation, data, window_data, index, sample_num)
        acce_operation.clean_partly()
        # window_data = np.array([[0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)],
        #                         [0 for i in range(WINDOW_SIZE)]])
        # index = 0
        # sample_num = 0
        # acce_operation.clean()