import datetime

import numpy as np
import time
from acce_operation import Acce_operation
import argparse

debug_No = 0
WINDOW_SIZE = 1500
SAMPLE_SIZE = 400
SPLIT_NUM = 7500


class Provider(object):
    def __init__(self, tag, data_dir):
        self.tag = tag
        raw_data_lines = open(data_dir).readlines()
        self.data = [[], [], []]
        self.data_size = len(raw_data_lines)
        # raw_data_lines = raw_data_lines[self.data_size // 3:]
        # self.data_size = len(raw_data_lines)
        for data_str in raw_data_lines:
            acc_raw_data = [int(x) for x in data_str.split()]
            self.data[0].append(acc_raw_data[1])
            self.data[1].append(acc_raw_data[2])
            self.data[2].append(acc_raw_data[3])

    def __call__(self, *args, **kwargs):
        for i in range(self.data_size):
            yield [self.data[j][i] for j in range(3)]

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


def run(acce_operation, provider):
    window_data = np.array([[0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)],
                            [0 for i in range(WINDOW_SIZE)]])
    index = 0
    sample_num = 0
    for data in provider():
        for i in range(3):
            window_data[i][index] = data[i]
        index += 1
        sample_num += 1
        if index == WINDOW_SIZE:
            index = 0
        if sample_num == SAMPLE_SIZE:
            sample_num = 0
            ret_data = np.concatenate([window_data[:, index:], window_data[:, :index]], axis=1)
            operation(ret_data, acce_operation)
            # time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('tag', type=int)
    parser.add_argument('--model_dir', type=str, default="../model/rnn_1")
    args = parser.parse_args()
    model_dir = args.model_dir
    data_dir = args.data_dir
    tag = args.tag
    acce_operation = Acce_operation(model_dir)
    provider = Provider(tag, data_dir)
    run(acce_operation, provider)
