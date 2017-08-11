import serial
import datetime

import matplotlib.pyplot as pl
import numpy as np
import time

a = serial.Serial('/dev/ttyACM0', 115200)
WINDOW_SIZE = 1500
SAMPLE_SIZE = 400

fig = pl.figure("ACCE")
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim([10, 100])
ax.set_ylim([0, 120000])
lines = []


def plot(data):
    x = np.array(range(len(data[0]) // 2 + 1)) / len(data[0]) * 4000
    x_freq = abs(np.fft.rfft(np.array(data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(data[2], dtype=np.float32)))
    freqs = [x_freq, y_freq, z_freq]
    for i in range(3):
        lines[i].set_xdata(x[1:])
        lines[i].set_ydata(freqs[i][1:])

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
    a.write(b'\x01')
    ret_data = [[0 for i in range(WINDOW_SIZE * 2)], [0 for i in range(WINDOW_SIZE * 2)],
                [0 for i in range(WINDOW_SIZE * 2)]]
    pl.show()
    index = 0
    sample_num = 0
    for i in range(3):
        line, = ax.plot(ret_data[i])
        lines.append(line)
    while True:
        data = a.read(6)
        # print(data)
        num = []
        for i in range(3):
            num.append(int(data[i * 2] << 8 | data[i * 2 + 1]))
            if num[i] > 2 ** 15:
                num[i] -= 2 ** 16
        for i in range(3):
            ret_data[i][index] = num[i]
        if index >= WINDOW_SIZE:
            for i in range(3):
                ret_data[i][index - WINDOW_SIZE] = num[i]

        index += 1
        sample_num += 1
        if index == WINDOW_SIZE * 2:
            index = WINDOW_SIZE
        if sample_num == SAMPLE_SIZE:
            sample_num = 0
            if index < WINDOW_SIZE:
                continue
            window_data = [ret_data[0][index - WINDOW_SIZE: index], ret_data[1][index - WINDOW_SIZE: index],
                           ret_data[2][index - WINDOW_SIZE: index]]
            plot(window_data)


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
