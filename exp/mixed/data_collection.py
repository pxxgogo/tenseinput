import os
import json
import numpy as np
from mix_model import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

EMG_IN_CHANNELS = 6
EMG_IN_SIZE = 50
ACC_IN_CHANNELS = 3
ACC_IN_SIZE = 251
INTERVAL_TIME = 10



def fft(window_data):
    x_freq = abs(np.fft.rfft(np.array(window_data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(window_data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(window_data[2], dtype=np.float32)))
    freqs = [x_freq, y_freq, z_freq]
    return freqs

class Data_collection(object):
    def __init__(self, model_dir):
        config_dir = os.path.join(model_dir, "model_config.json")
        self.config = json.load(open(config_dir))
        self.config["batch_size"] = 1
        session_config = tf.ConfigProto(log_device_placement=False)
        session_config.gpu_options.allow_growth = False
        self.session = tf.Session(config=session_config)
        with tf.variable_scope("model", reuse=None):
            self.model = Model(config=self.config)
        self.session.run(tf.global_variables_initializer())
        new_saver = tf.train.Saver()
        new_saver.restore(self.session, tf.train.latest_checkpoint(
            model_dir))
        self.sample_index = 0
        self.emg_data_index = 0
        self.emg_data = np.zeros([EMG_IN_CHANNELS, EMG_IN_SIZE])
        self.acc_data_index = 0
        self.acc_data = np.zeros([ACC_IN_CHANNELS, ACC_IN_SIZE])

        self.fig = plt.figure()
        self.fig.patches.append(mpatches.Circle([0.5, 0.5], 0.1, transform=fig.transFigure))
        self.result_pointer = 0
        self.result_window = np.zeros(1)


    def update_result_window(self, ret):
        self.result_window[self.result_pointer] = ret
        ret_raw = np.mean(self.result_window)
        if ret_raw > 0:
            ret_raw = 1
        self.result_pointer += 1
        if self.result_pointer == self.result_window.shape[0]:
            self.result_pointer = 0
        return ret_raw

    def draw_ret(self, ret, logits):
        if not ret == -1:
            ret = self.update_result_window(ret)
            if ret == 0:
                self.fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=self.fig.transFigure, color=(ret, 0, 0)))
            else:
                self.fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=self.fig.transFigure, color=(ret, 0, 0)))
            self.fig.show()
            plt.pause(0.05)
            self.fig.clf()

    def predict(self):
        acc_data = np.concatenate([self.acc_data[:, self.acc_data_index:], self.acc_data[:, :self.acc_data_index]], axis=1)
        emg_data = np.concatenate([self.emg_data[:, self.emg_data_index:], self.emg_data[:, :self.emg_data_index]], axis=1)
        # print(acc_data)
        acc_fft_data = np.array([fft(acc_data)])
        emg_data = np.array([emg_data])
        fetches = [self.model.predict_op, self.model.logits]
        feed_dict = {}
        feed_dict[self.model.acc_input_data] = acc_fft_data
        feed_dict[self.model.emg_input_data] = emg_data
        predict_val, logits = self.session.run(fetches, feed_dict)
        print(predict_val, logits, logits[0][0] - logits[0][1])
        self.draw_ret(predict_val, logits)


    def feed_emg_data(self, data):
        self.emg_data[:, self.emg_data_index] = np.array(data)
        self.sample_index += 1
        if self.emg_data_index >= EMG_IN_SIZE:
            self.emg_data_index = 0
        if self.sample_index == INTERVAL_TIME:
            self.predict()
            self.sample_index = 0

    def feed_acc_data(self, data):
        self.acc_data[:, self.acc_data_index] = np.array(data)
        if self.acc_data_index >= ACC_IN_SIZE:
            self.acc_data_index = 0
