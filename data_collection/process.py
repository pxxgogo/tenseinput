import datetime
import time

import serial

import numpy as np

from estimator import Estimator

from ctypes import cdll
from ctypes import c_double


def read_data(data, offset):
    arr = []
    for i in range(3):
        arr.append(int(data[i * 2 + offset] << 8 | data[i * 2 + 1 + offset]))
        if arr[i] > 2 ** 15:
            arr[i] -= 2 ** 16
    return arr


a = serial.Serial('/dev/ttyACM0', 115200)

raw_datafile = open(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '-raw.txt', 'w')
datafile = open(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.txt', 'w')

a.write(b'\x01')

count = 0
lastTp = time.time()

est = Estimator()

while True:
    # read data
    data = a.read(14)
    raw_acc = read_data(data, 0)
    raw_gyr = read_data(data, 8)

    # time interval between two samples (sec)
    tp = time.time()
    ts = round(tp * 1000)
    dt = tp - lastTp
    lastTp = tp

    # attitude estimation
    vp = est.feed_data(dt, raw_gyr, raw_acc)

    raw_datafile.write(str(ts) + ' ' + str(raw_gyr[1]) + ' ' + str(raw_gyr[2]) + ' ' + str(raw_acc[0]) + ' ' + str(raw_acc[1]) + ' ' + str(raw_acc[2]) + ' ' + str(raw_gyr[0]) + '\n')
    datafile.write(str(ts) + ' ' + str(vp[0]) + ' ' + str(vp[1]) + ' ' + str(vp[2]) + '\n')

    count += 1

    if count % 80 == 0:
        count = 0
        # print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (acc[0], acc[1], acc[2], gyr[0], gyr[1], gyr[2], q[0], q[1], q[2], q[3]),
            # end='\r')
        
        print(vp, end='\r')

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
