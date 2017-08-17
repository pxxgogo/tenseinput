import serial

import matplotlib.pyplot as pl
import numpy as np
import time
from estimator import Estimator


a = serial.Serial('/dev/ttyACM0', 115200)
WINDOW_SIZE = 400
SAMPLE_SIZE = 100

fig = pl.figure("ACCE")
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim([10, 100])
ax.set_ylim([0, 12000])
lines = []


def read_data(data, offset):
    arr = []
    for i in range(3):
        arr.append(int(data[i * 2 + offset] << 8 | data[i * 2 + 1 + offset]))
        if arr[i] >= 2 ** 15:
            arr[i] -= 2 ** 16
    return arr


def plot(data):
    x = np.array(range(len(data[0]) // 2 + 1)) / len(data[0]) * 1000
    x_freq = abs(np.fft.rfft(np.array(data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(data[2], dtype=np.float32)))
    freqs = np.array([x_freq, y_freq, z_freq])
    for i in range(3):
        # print(x[1:].shape, data[i][1:].shape)
        lines[i].set_xdata(x[1:])
        # lines[i].set_ydata(freqs[i][1:])
        lines[i].set_ydata(abs(data[i][1:][:200]))
    # print(freqs[:, 50], end="\r")
    fig.canvas.draw()
    fig.canvas.flush_events()

    # # print(x_freq)
    # pl.axis([10,300,0,120000])
    # pl.plot(x[1:], x_freq[1:], 'r')
    # # pl.draw()
    # # pl.pause(0.001)

    # pl.plot(x[1:], y_freq[1:], 'b')
    # # pl.draw()
    # # pl.pause(0.001)

    # pl.plot(x[1:], z_freq[1:], 'k')
    # pl.draw()
    pl.pause(0.001)
    # pl.clf()


def run():
    pl.ion()
    pl.show()
    est = Estimator()
    a.write(b'\x01')
    window_data = np.array([[0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)],
                [0 for i in range(WINDOW_SIZE)]])
    pl.show()
    index = 0
    sample_num = 0
    lastTp = time.time()
    for i in range(3):
        line, = ax.plot(window_data[i])
        lines.append(line)
    while True:
        data = a.read(14)
        raw_acc = read_data(data, 0)
        raw_gyr = read_data(data, 8)

        tp = time.time()
        dt = tp - lastTp
        lastTp = tp

        # attitude estimation
        vp = est.feed_data(dt, raw_gyr, raw_acc)
        print(vp, end="\r")
        for i in range(3):
            window_data[i][index] = vp[i]*4096

        index += 1
        sample_num += 1
        if index == WINDOW_SIZE:
            index = 0
        if sample_num == SAMPLE_SIZE:
            sample_num = 0
            ret_data = np.concatenate([window_data[:, index:], window_data[:, :index]], axis=1)
            plot(ret_data)

run()

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
