import argparse
import json

import numpy as np

WINDOW_SIZE = 1500
SAMPLE_SIZE = 400
SPLIT_WINDOW_SIZE = 10


def fft(window_data):
    x_freq = abs(np.fft.rfft(np.array(window_data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(window_data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(window_data[2], dtype=np.float32)))
    freqs = [x_freq, y_freq, z_freq]
    return freqs


def fft_process(acc_data):
    window_data = np.array([[0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)]])
    acc_index = 0
    index = 0
    sample_num = 0
    acc_fft_data = []
    acc_data_length = len(acc_data[0])
    fft_num = 0
    while acc_index < acc_data_length:
        num = [acc_data[i][acc_index] for i in range(3)]
        acc_index += 1
        for i in range(3):
            window_data[i, index] = num[i]
        index += 1
        sample_num += 1
        if index == WINDOW_SIZE:
            index = 0
        if sample_num == SAMPLE_SIZE:
            sample_num = 0
            ret_data = np.concatenate([window_data[:, index:], window_data[:, :index]], axis=1)
            fft_data = fft(ret_data)
            acc_fft_data.append(fft_data)
            fft_num += 1
            # if fft_num % 1000 == 0:
            #     print(fft_num)
    return acc_fft_data


def split_data(acc_fft_data):
    acc_fft_data = np.array(acc_fft_data)
    new_acc_fft_data = []
    num = acc_fft_data.shape[0] // SPLIT_WINDOW_SIZE
    for i in range(num):
        new_acc_fft_data.append(acc_fft_data[i * SPLIT_WINDOW_SIZE: (i + 1) * SPLIT_WINDOW_SIZE])
    new_acc_fft_data = np.array(new_acc_fft_data)
    return new_acc_fft_data


def prune_data(acc_fft_data):
    new_acc_fft_data = []
    for sub_fft_data in acc_fft_data:
        if len(sub_fft_data) < 10:
            continue
        new_acc_fft_data.append(sub_fft_data[:10])
    new_acc_fft_data = np.array(new_acc_fft_data)
    return new_acc_fft_data


def data_process(acc_file_name, tag):
    with open(acc_file_name) as acc_file_handle:
        acc_data = json.loads(acc_file_handle.read())
    if tag == 0:
        acc_sub_data = acc_data[0]
        acc_fft_data = fft_process(acc_sub_data)
        # print(len(acc_fft_data))
        acc_fft_data = split_data(acc_fft_data)
        return acc_fft_data
    else:
        acc_fft_data = []
        for acc_sub_data in acc_data:
            sub_fft_data = fft_process(acc_sub_data[0])
            # print(len(sub_fft_data[0]))
            acc_fft_data.append(sub_fft_data)
        acc_fft_data = prune_data(acc_fft_data)
        return acc_fft_data


def export(acc_fft_data, output_dir):
    # print(acc_fft_data.shape)
    np.save(output_dir, acc_fft_data)
    # acc_fft_data.(output_dir, " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', type=int)
    parser.add_argument('acc', type=str)
    parser.add_argument('--output_dir', type=str, default="./acc_fft_data")
    args = parser.parse_args()
    acc_file = args.acc
    tag = args.tag
    output_dir = args.output_dir
    acc_fft_data = data_process(acc_file, tag)
    print(acc_fft_data.shape)
    export(acc_fft_data, output_dir)


main()
