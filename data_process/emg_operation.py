import argparse
import json
import numpy as np
import random

PADDING = 10
SPLIT_WINDOW_SIZE = 30
MAX_DATA_NUM = 1300


def prune_data(emg_raw_data):
    emg_data_list = []
    for data in emg_raw_data:
        tag = data[1]
        emg_pruned_data = []
        flag = True
        for i in range(6):
            x = data[0][i][PADDING:]
            x = x[:SPLIT_WINDOW_SIZE]
            if not len(x) == SPLIT_WINDOW_SIZE:
                flag = False
        if not flag:
            continue
        for i in range(6):
            x = data[0][i][PADDING:]
            x = x[:SPLIT_WINDOW_SIZE]
            emg_pruned_data.append(x[:SPLIT_WINDOW_SIZE])
        emg_data_list.append([emg_pruned_data, tag])
    return emg_data_list


def split_data(emg_raw_data):
    emg_data = np.array(emg_raw_data[0])
    new_emg_data = []
    num = emg_data.shape[1] // SPLIT_WINDOW_SIZE
    for i in range(num):
        new_emg_data.append([emg_data[:, i * SPLIT_WINDOW_SIZE: (i + 1) * SPLIT_WINDOW_SIZE], 0])
    random.shuffle(new_emg_data)
    new_emg_data = np.array(new_emg_data[:MAX_DATA_NUM])
    return new_emg_data


def fft(window_data):
    freqs = []
    for sub_data in window_data:
        freq = abs(np.fft.rfft(np.array(sub_data, dtype=np.float32)))
        freqs.append(freq)
    return freqs


def fft_process(emg_data_list):
    emg_fft_data_list = []
    for data in emg_data_list:
        tag = data[1]
        fft_data = fft(data[0])
        emg_fft_data_list.append([fft_data, tag])
    return emg_fft_data_list


def post_operate(emg_data_list):
    final_data_list = []
    for data in emg_data_list:
        tag = data[1]
        sub_data = data[0]
        ret = []
        for x in sub_data:
            ret.extend(x)
        final_data_list.append([ret, tag])
    return final_data_list


def data_process(emg_file_name, fft_flag, tag):
    emg_raw_data_list = json.load(open(emg_file_name))
    emg_operated_data_list = []
    if tag > 0:
        emg_data_list = prune_data(emg_raw_data_list)
    else:
        emg_data_list = split_data(emg_raw_data_list)
    if fft_flag == 1:
        emg_data_list = fft_process(emg_data_list)
    final_data_list = post_operate(emg_data_list)
    return np.array(final_data_list)

def export(emg_operated_data, output_dir):
    # for data in emg_operated_data:
    #     print(len(data[0]))
    print(emg_operated_data.shape)
    np.save(output_dir, emg_operated_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', type=int)
    parser.add_argument('emg', type=str)
    parser.add_argument('--fft', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="./emg_fft_data")
    args = parser.parse_args()
    tag = args.tag
    emg_file_name = args.emg
    fft_flag = args.fft
    output_dir = args.output_dir
    emg_operated_data = data_process(emg_file_name, fft_flag, tag)
    export(emg_operated_data, output_dir)

main()
