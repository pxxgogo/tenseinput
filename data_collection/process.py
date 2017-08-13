import serial
import time
import datetime

a = serial.Serial('COM10', 115200)

datafile = open(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')+'.txt', 'w')

a.write(b'\x01')

count = 0
while True:
    data = a.read(6)
    num = []
    for i in range(3):
        num.append(int(data[i*2]<<8 | data[i*2+1]))
        if num[i] > 2**15:
            num[i] -= 2**16
    tp = int(round(time.time()*1000))
    datafile.write(str(tp)+' '+str(num[0])+' '+str(num[1])+' '+str(num[2])+'\n')
    count += 1
    if count % 80 == 0:
    	count = 0
    	print(str(num[0])+'\t'+str(num[1])+'\t'+str(num[2]), end='\r')


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
