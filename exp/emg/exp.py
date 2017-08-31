import serial

import numpy as np
import time

from emg_operation import Emg_operation
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import win32file

# configure pipe
fileHandle = win32file.CreateFile("\\\\.\\pipe\\emg_pipe",
									win32file.GENERIC_READ | win32file.GENERIC_WRITE,
									0, None, win32file.OPEN_EXISTING, 0 , None)

SAMPLE_RATE = 50
CHANNELS = 8

fig = plt.figure()
fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure))

debug_No = 0
result_window = np.zeros(1)
result_pointer = 0

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


def run(emg_operation):
    while True:
        ret, s_bytes = win32file.ReadFile(fileHandle, 10240)
        s = s_bytes.decode("utf-8")
        if ret != 0:
            print("Error reading from pipe")
            exit(1)
        if len(s) == 0:
            time.sleep(0.01)
            continue

        for line in s.strip().split('\n'):
            if line.startswith('\0'):
                line = line[1:]
            row = line.strip().split(' ')
            rowdata = []
            for i, item in enumerate(row):
                if i == 3 or i == 4:
                    continue
                rowdata.append(int(item))
            # print(rowdata)
            ret, logits = emg_operation.feed_data(rowdata)
            operation(ret, logits)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="../../model/0828/emg/time_seq/model_3_layer_1")
    args = parser.parse_args()
    model_dir = args.model_dir
    emg_operation = Emg_operation(model_dir)
    run(emg_operation)
