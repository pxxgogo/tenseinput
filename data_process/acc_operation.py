import argparse
import json

import numpy as np
import random

WINDOW_SIZE = 300
SAMPLE_SIZE = 300
SPLIT_WINDOW_SIZE = 2
PADDING_FRAME = 0
PADDING_TIME = 200
PRUNE_SAMPLE_TIMES = 2


def fft(window_data):
    x_freq = abs(np.fft.rfft(np.array(window_data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(window_data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(window_data[2], dtype=np.float32)))
    freqs = [x_freq, y_freq, z_freq]
    return freqs


def fft_process_sliding_windows(acc_data):
    window_data = np.array(
        [[0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)], [0 for i in range(WINDOW_SIZE)]])
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
            print(fft_data[0][0])
            acc_fft_data.append(fft_data)
            fft_num += 1
            # if fft_num % 1000 == 0:
            #     print(fft_num)
    return acc_fft_data


def fft(window_data):
    freqs = []
    for sub_data in window_data:
        freq = abs(np.fft.rfft(np.array(sub_data, dtype=np.float32)))
        freqs.append(freq)
    # print(freqs)
    return np.array(freqs)


def fft_process(data_list, flag=0):
    fft_data_list = []
    for data in data_list:
        if flag == 0:
            fft_data = []
            for sub_data in data[0]:
                fft_data.append(fft(sub_data))
            try:
                fft_data = np.array(fft_data)
            except Exception as e:
                print(fft_data[0].shape, fft_data[1].shape)
                print(e)
                exit()

        else:
            fft_data = fft(data[0])
        fft_data_list.append([fft_data, data[1], data[2]])
    return fft_data_list


def prune_data(acc_data, window_size, padding_time=0, flag=0):
    new_acc_data = []
    if flag == 0:
        for sub_data in acc_data:
            if len(sub_data[0][0]) < window_size:
                continue
            acc_pruned_data_p = []
            flag = True
            for t in range(PRUNE_SAMPLE_TIMES):
                acc_pruned_data = []
                for i in range(3):
                    x = sub_data[0][i][padding_time + t * window_size: padding_time + (t + 1) * window_size]
                    if len(x) < window_size:
                        flag = False
                        break
                if not flag:
                    break
                for i in range(3):
                    x = sub_data[0][i][padding_time + t * window_size: padding_time + (t + 1) * window_size]
                    acc_pruned_data.append(x)
                acc_pruned_data_p.append(acc_pruned_data)
            if not flag:
                continue
            new_acc_data.append([np.array(acc_pruned_data_p), sub_data[1], sub_data[2]])
    elif flag == 1:
        for sub_data in acc_data:
            if len(sub_data[0][0]) < window_size:
                continue
            for t in range(PRUNE_SAMPLE_TIMES):
                acc_pruned_data = []
                for i in range(3):
                    x = sub_data[0][i][padding_time + t * window_size: padding_time + (t + 1) * window_size]
                    acc_pruned_data.append(x)
                new_acc_data.append([np.array(acc_pruned_data), sub_data[1], sub_data[2]])
    else:
        for sub_data in acc_data:
            if len(sub_data[0][0]) < window_size:
                continue
            acc_pruned_data = [[] for i in range(PRUNE_SAMPLE_TIMES * 3)]
            for t in range(PRUNE_SAMPLE_TIMES):
                for i in range(3):
                    x = sub_data[0][i][padding_time + t * window_size: padding_time + (t + 1) * window_size]
                    acc_pruned_data[i * PRUNE_SAMPLE_TIMES + t] = np.array(x)
            new_acc_data.append([acc_pruned_data, sub_data[1], sub_data[2]])
    return new_acc_data


def post_operate(data_list, flag=0):
    final_data_list = []
    if flag == 0:
        for data in data_list:
            sub_data = data[0]
            new_sub_data = []
            for sub_sub_data in sub_data:
                ret = []
                for x in sub_sub_data:
                    ret.extend(x)
                new_sub_data.append(ret)
            final_data_list.append([np.array(new_sub_data), data[1], data[2]])
    else:
        for data in data_list:
            sub_data = data[0]
            ret = []
            for x in sub_data:
                ret.extend(x)
            final_data_list.append([np.array(ret), data[1], data[2]])
    return final_data_list


def data_process(file_name, fft_type, data_format):
    with open(file_name) as file_handle:
        data = json.loads(file_handle.read())
    data = prune_data(data, WINDOW_SIZE, PADDING_TIME, flag=fft_type)
    fft_data = fft_process(data, flag=fft_type)
    if data_format == 1:
        return post_operate(fft_data, flag=fft_type)
    else:
        return fft_data


def export(acc_fft_data, output_dir):
    acc_fft_data = np.array(acc_fft_data)
    print(acc_fft_data.shape, np.array(list(acc_fft_data[0][0])).shape)

    np.save(output_dir, acc_fft_data)
    # acc_fft_data.(output_dir, " ")


# fft_type:
#  0: sliding
#  1: static
# data_format
#  0: channels
#  1: joint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('acc', type=str)
    parser.add_argument('--fft_type', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="./acc_fft_data")
    parser.add_argument('--data_format', type=int, default=0)
    args = parser.parse_args()
    acc_file_name = args.acc
    fft_type = args.fft_type
    output_dir = args.output_dir
    data_format = args.data_format
    acc_fft_data = data_process(acc_file_name, fft_type, data_format)
    export(acc_fft_data, output_dir)


main()
