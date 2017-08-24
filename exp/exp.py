import serial

import numpy as np
import time
from estimator import Estimator

from acce_operation import Acce_operation
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig = plt.figure()
fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure))

debug_No = 0
a = serial.Serial('/dev/ttyACM0', 115200)
WINDOW_SIZE = 400
SAMPLE_SIZE = 100


def operation(data, acce_operation):
    x = np.array(range(len(data[0]) // 2 + 1)) / len(data[0]) * 4000
    x_freq = abs(np.fft.rfft(np.array(data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(data[2], dtype=np.float32)))
    freqs = np.array([x_freq, y_freq, z_freq])
    ret, debug_val = acce_operation.feed_data(freqs)
    global debug_No
    if not ret == -1:
        logits = debug_val[0]
        # if logits[2] > 2.5:
        #     ret = 2
        # elif logits[1] > 2.5:
        #     ret = 1
        # else:
        #     ret = 0
        print(debug_No, ret, logits)
        if ret == 0:
            fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure, fc="Grey"))
        else:
            fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure, fc="Red"))
        fig.show()
        plt.pause(0.05)
        fig.clf()
        debug_No += 1


def read_data(data, offset):
    arr = []
    for i in range(3):
        arr.append(int(data[i * 2 + offset] << 8 | data[i * 2 + 1 + offset]))
        if arr[i] >= 2 ** 15:
            arr[i] -= 2 ** 16
    return arr


def run(acce_operation):
    est = Estimator()

    a.write(b'\x01')
    window_data = np.array([[0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)],
                            [0 for i in range(WINDOW_SIZE)]])
    index = 0
    sample_num = 0
    lastTp = time.time()

    while True:
        data = a.read(14)
        raw_acc = read_data(data, 0)
        raw_gyr = read_data(data, 8)

        # print(data)

        tp = time.time()
        dt = tp - lastTp
        lastTp = tp

        # attitude estimation
        vp = est.feed_data(dt, raw_gyr, raw_acc)
        # print(vp, end="\r")
        for i in range(3):
            window_data[i][index] = vp[i] * 4096

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
    parser.add_argument('--model_dir', type=str, default="../model/0816/rnn_1")
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
