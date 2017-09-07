import serial

import numpy as np
import time
from estimator import Estimator

from acc_static_operation import Acce_operation
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors

fig = plt.figure()
fig.patches.append(mpatches.Circle([0.5, 0.5], 0.1, transform=fig.transFigure))

debug_No = 0
a = serial.Serial('/dev/ttyACM0', 115200)
result_window = np.zeros(1)
result_pointer = 0
TIMES = 1


def update_result_window(ret):
    global result_pointer
    global result_window
    result_window[result_pointer] = ret
    ret_raw = np.mean(result_window)
    result_pointer += 1
    if result_pointer == result_window.shape[0]:
        result_pointer = 0
    return ret_raw

def operation(ret, logits):
    global debug_No
    if not ret == -1:
        # if logits[2] > 2.5:
        #     ret = 2
        # elif logits[1] > 2.5:
        #     ret = 1
        # else:
        #     ret = 0
        ret = update_result_window(ret)
        if ret == 0:
            fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure, color=(ret, 0, 0)))
        else:
            fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure, color=(ret, 0, 0)))
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


def run(acc_operation):
    est = Estimator()

    a.write(b'\x01')
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
            vp[i] = vp[i] * TIMES
        # print(vp)

        ret, logits = acc_operation.feed_data(vp)
        operation(ret, logits)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="../../model/0828/emg/time_seq/model_3_layer_1")
    args = parser.parse_args()
    model_dir = args.model_dir
    acc_operation = Acce_operation(model_dir)
    run(acc_operation)
