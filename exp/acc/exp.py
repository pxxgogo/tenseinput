import serial

import numpy as np
import time
from estimator import Estimator
import os
import json

from acce_operation import Acce_operation
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors

fig = plt.figure()
fig.patches.append(mpatches.Circle([0.5, 0.5], 0.1, transform=fig.transFigure))

debug_No = 0
a = serial.Serial('/dev/ttyACM0', 115200)
WINDOW_SIZE = 500
SEQUENCE_SIZE = 1
TOTAL_WINDOW_SIZE = SEQUENCE_SIZE * WINDOW_SIZE
SAMPLE_SIZE = 100
result_window = np.zeros(1)
result_pointer = 0


def update_result_window(ret):
    global result_pointer
    global result_window
    result_window[result_pointer] = ret
    ret_raw = np.mean(result_window)
    if ret_raw > 0:
        ret_raw = 1
    result_pointer += 1
    if result_pointer == result_window.shape[0]:
        result_pointer = 0
    return ret_raw


def fft(window_data):
    x_freq = abs(np.fft.rfft(np.array(window_data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(window_data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(window_data[2], dtype=np.float32)))
    freqs = [x_freq, y_freq, z_freq]
    return freqs

def read_data(data, offset):
    arr = []
    for i in range(3):
        arr.append(int(data[i * 2 + offset] << 8 | data[i * 2 + 1 + offset]))
        if arr[i] >= 2 ** 15:
            arr[i] -= 2 ** 16
    return arr

def draw_ret(ret, logits):
    global debug_No
    if not ret == -1:
        ret = update_result_window(ret)
        if ret == 0:
            fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure, color=(ret, 0, 0)))
        else:
            fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure, color=(ret, 0, 0)))
        fig.show()
        plt.pause(0.05)
        fig.clf()
        debug_No += 1


def operate(acce_operation, window_data, index):
    window_data = np.concatenate([window_data[:, index:], window_data[:, :index]], axis=1)
    # print(window_data)
    ret_data = []
    for i in range(SEQUENCE_SIZE):
        data = window_data[:, - (i + 1) * WINDOW_SIZE: window_data.shape[1] - i * WINDOW_SIZE]
        # data = window_data[:, i * WINDOW_SIZE: (i + 1) * WINDOW_SIZE]
        fft_data = fft(data)
        ret_data.append(fft_data)
    ret_data = np.array(ret_data)
    # print(ret_data)

    if SEQUENCE_SIZE != 1:
        ret_data = np.reshape(ret_data, [1, ret_data.shape[0], ret_data.shape[1], ret_data.shape[2]])
    predict_val, logits = acce_operation.predict(ret_data)
    # print(predict_val, logits)
    draw_ret(predict_val, logits)




def run(model_dir):
    acce_operation = Acce_operation(model_dir)
    est = Estimator()

    a.write(b'\x01')
    window_data = np.array([[0 for i in range(TOTAL_WINDOW_SIZE)], [0 for i in range(TOTAL_WINDOW_SIZE)],
                            [0 for i in range(TOTAL_WINDOW_SIZE)]], dtype=np.float32)
    index = 0
    sample_num = 0
    lastTp = time.time()

    while True:
        data = a.read(14)
        raw_acc = read_data(data, 0)
        raw_gyr = read_data(data, 8)

        tp = time.time()
        dt = tp - lastTp
        lastTp = tp

        # attitude estimation
        vp = est.feed_data(dt, raw_gyr, raw_acc)
        # print(vp, end="\r")
        for i in range(3):
            window_data[i][index] = vp[i]

        index += 1
        sample_num += 1
        if index == TOTAL_WINDOW_SIZE:
            index = 0
        if sample_num == SAMPLE_SIZE:
            sample_num = 0
            operate(acce_operation, window_data, index)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="../../key_model/0906/acc/model_outlier_2_interaction")
    args = parser.parse_args()
    model_dir = args.model_dir
    config_dir = os.path.join(model_dir, 'model_config.json')
    config = json.load(open(config_dir))
    run(model_dir)

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
