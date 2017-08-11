import serial
import datetime

import numpy as np
import time
from acce_operation import Acce_operation
import argparse

debug_No = 0
a = serial.Serial('/dev/ttyACM0', 115200)
WINDOW_SIZE = 1500
SAMPLE_SIZE = 400


def operation(data, acce_operation):
    x = np.array(range(len(data[0]) // 2 + 1)) / len(data[0]) * 4000
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


def run(acce_operation):
    a.write(b'\x01')
    window_data = np.array([[0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)],
                [0 for i in range(WINDOW_SIZE)]])
    index = 0
    sample_num = 0
    while True:
        data = a.read(6)
        # print(data)
        num = []
        for i in range(3):
            num.append(int(data[i * 2] << 8 | data[i * 2 + 1]))
            if num[i] > 2 ** 15:
                num[i] -= 2 ** 16
        for i in range(3):
            window_data[i][index] = num[i]
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
    parser.add_argument('--model_dir', type=str, default="../model/rnn_1")
    args = parser.parse_args()
    model_dir = args.model_dir
    acce_operation = Acce_operation(model_dir)
    run(acce_operation)

'''
连线
模块 -> 开发板
VCC     3.3
GND     G
SCLK    A5
SDI     A7
SDO     A6
INT     A1
NCS     A4
'''
