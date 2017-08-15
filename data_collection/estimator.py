import numpy as np

from ctypes import cdll
from ctypes import c_double

# calibration constants for X, Y, Z components
ACC_OFFSET = [-202.0281770191, -81.7801589269, 419.6444280899]
GYR_OFFSET = [-15.353925709, -13.0307386379, 13.6007070723]

ACC_SCALE = [0.0023891523, 0.0023878861, 0.0023579114]
GYR_SCALE = 0.001064225


class Estimator(object):

    def __init__(self):
        # load dynamic library
        self.lib = cdll.LoadLibrary('fusion/libest.so')


        # set ctypes type requirements
        self.lib.init_est()
        self.lib.update_est.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double]
        self.lib.q0.restype = c_double
        self.lib.q1.restype = c_double
        self.lib.q2.restype = c_double
        self.lib.q3.restype = c_double



    def feed_data(self, dt, raw_gyr, raw_acc):

        # calibrate and convert to SI units
        acc = [0.0, 0.0, 0.0]
        gyr = [0.0, 0.0, 0.0]
        for i in range(3):
            acc[i] = float(raw_acc[i]) + ACC_OFFSET[i]
            acc[i] *= ACC_SCALE[i]
            gyr[i] = float(raw_gyr[i]) + GYR_OFFSET[i]
            gyr[i] *= GYR_SCALE

        # attitude estimation
        self.lib.update_est(dt, gyr[0], gyr[1], gyr[2], acc[0], acc[1], acc[2])
        
        q = [0, 0, 0, 0]

        q[0] = self.lib.q0()
        q[1] = self.lib.q1()
        q[2] = self.lib.q2()
        q[3] = self.lib.q3()

        # rotate unit vector by quarternion
        v = np.array(acc)
        u = np.array(q[1:])
        s = q[0]
        vp = 2.0 * np.dot(u,v) * u + (s**2 - np.dot(u,u)) * v + 2.0 * s * np.cross(u,v) - np.array([0,0,9.8])

        return vp