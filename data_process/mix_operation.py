import argparse
import json
import numpy as np
import random

EMG_PADDING = 10
EMG_SPLIT_WINDOW_SIZE = 30

ACC_PADDING = 200
ACC_SPLIT_WINDOW_SIZE = 600
MAX_DATA_NUM = 1300


def prune_data(raw_data):
    data_list = []
    for data in raw_data:
        emg_data = data[0][1]
        acc_data = data[0][0]
        emg_pruned_data = []
        flag = True
        for i in range(6):
            x = emg_data[i][-EMG_SPLIT_WINDOW_SIZE:]
            if not len(x) == EMG_SPLIT_WINDOW_SIZE:
                flag = False
        if not flag:
            continue
        for i in range(6):
            x = emg_data[i][-EMG_SPLIT_WINDOW_SIZE:]
            emg_pruned_data.append(x[:EMG_SPLIT_WINDOW_SIZE])

        acc_pruned_data = []
        flag = True
        for i in range(3):
            x = acc_data[i][-ACC_SPLIT_WINDOW_SIZE:]
            if not len(x) == ACC_SPLIT_WINDOW_SIZE:
                flag = False
        if not flag:
            continue
        for i in range(3):
            x = acc_data[i][-ACC_SPLIT_WINDOW_SIZE:]
            acc_pruned_data.append(x[:ACC_SPLIT_WINDOW_SIZE])
        data_list.append([[acc_pruned_data, emg_pruned_data], data[1], data[2]])
    return data_list


def fft(window_data):
    freqs = []
    for sub_data in window_data:
        freq = abs(np.fft.rfft(np.array(sub_data, dtype=np.float32)))
        freqs.append(freq)
    return freqs


def fft_process(data_list):
    fft_data_list = []
    for data in data_list:
        acc_data = data[0][0]
        emg_data = data[0][1]
        acc_fft_data = fft(acc_data)
        fft_data_list.append([[acc_fft_data, emg_data], data[1], data[2]])
    return fft_data_list


def data_process(data_file_name):
    raw_data_list = json.load(open(data_file_name))
    data_list = prune_data(raw_data_list)
    data_list = fft_process(data_list)
    return np.array(data_list)


def export(operated_data, output_dir):
    # for data in operated_data:
    #     print(len(data[0]))
    print(operated_data.shape)
    np.save(output_dir, operated_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--output_dir', type=str, default="./emg_fft_data")
    args = parser.parse_args()
    data_file_name = args.data
    output_dir = args.output_dir
    operated_data = data_process(data_file_name)
    export(operated_data, output_dir)


main()
