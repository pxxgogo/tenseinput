import argparse
import json
import numpy as np
import random

EMG_SPLIT_WINDOW_SIZE = 25

ACC_SPLIT_WINDOW_SIZE = 500

CRITICAL_TIMES = 2
CRITICAL_GESTURES = []

PRUNE_SAMPLE_TIMES = 1


def prune_data(raw_data, sample_flag):
    data_list = []
    if sample_flag == 0:
        for data in raw_data:
            emg_data = data[0][1]
            acc_data = data[0][0]
            gesture_No = data[2]
            if gesture_No in CRITICAL_GESTURES:
                critical_times = CRITICAL_TIMES
                # print(gesture_No)
            else:
                critical_times = 1
            for k in range(critical_times):
                emg_pruned_data = []
                flag = True
                for i in range(6):
                    # print(len(emg_data[i]))
                    x = emg_data[i][-(k + 1) * EMG_SPLIT_WINDOW_SIZE: len(emg_data[i]) - k * EMG_SPLIT_WINDOW_SIZE]
                    if not len(x) == EMG_SPLIT_WINDOW_SIZE:
                        flag = False
                if not flag:
                    continue
                for i in range(6):
                    x = emg_data[i][-(k + 1) * EMG_SPLIT_WINDOW_SIZE: len(emg_data[i]) - k * EMG_SPLIT_WINDOW_SIZE]
                    emg_pruned_data.append(x)

                acc_pruned_data = []
                flag = True
                for i in range(3):
                    x = acc_data[i][-(k + 1) * ACC_SPLIT_WINDOW_SIZE: len(acc_data[i]) - k * ACC_SPLIT_WINDOW_SIZE]
                    if not len(x) == ACC_SPLIT_WINDOW_SIZE:
                        flag = False
                if not flag:
                    continue
                for i in range(3):
                    x = acc_data[i][-(k + 1) * ACC_SPLIT_WINDOW_SIZE: len(acc_data[i]) - k * ACC_SPLIT_WINDOW_SIZE]
                    acc_pruned_data.append(x)
                data_list.append([[acc_pruned_data, emg_pruned_data], data[1], data[2]])
    else:
        # print(raw_data)
        for data in raw_data:
            emg_data = data[0][1]
            acc_data = data[0][0]
            gesture_No = data[2]
            flag = True
            sum_emg_data = []
            sum_acc_data = []
            for k in range(PRUNE_SAMPLE_TIMES):
                emg_pruned_data = []
                for i in range(6):
                    # print(len(emg_data[i]))
                    x = emg_data[i][-(k + 1) * EMG_SPLIT_WINDOW_SIZE: len(emg_data[i]) - k * EMG_SPLIT_WINDOW_SIZE]
                    if not len(x) == EMG_SPLIT_WINDOW_SIZE:
                        flag = False
                        break
                if not flag:
                    break
                for i in range(6):
                    x = emg_data[i][-(k + 1) * EMG_SPLIT_WINDOW_SIZE: len(emg_data[i]) - k * EMG_SPLIT_WINDOW_SIZE]
                    emg_pruned_data.append(x)
                sum_emg_data.append(emg_pruned_data)
            if not flag:
                continue
            for k in range(PRUNE_SAMPLE_TIMES):
                acc_pruned_data = []
                flag = True
                for i in range(3):
                    x = acc_data[i][-(k + 1) * ACC_SPLIT_WINDOW_SIZE: len(acc_data[i]) - k * ACC_SPLIT_WINDOW_SIZE]
                    if not len(x) == ACC_SPLIT_WINDOW_SIZE:
                        flag = False
                        break
                if not flag:
                    break
                for i in range(3):
                    x = acc_data[i][-(k + 1) * ACC_SPLIT_WINDOW_SIZE: len(acc_data[i]) - k * ACC_SPLIT_WINDOW_SIZE]
                    acc_pruned_data.append(x)
                sum_acc_data.append(acc_pruned_data)
            if not flag:
                continue
            data_list.append([[sum_acc_data, sum_emg_data], data[1], data[2]])
    return data_list


def fft(window_data):
    freqs = []
    for sub_data in window_data:
        freq = list(abs(np.fft.rfft(np.array(sub_data, dtype=np.float32))))
        freqs.append(freq)
    return freqs


def fft_process(data_list, sample_flag):
    fft_data_list = []
    for data in data_list:
        acc_data = data[0][0]
        emg_data = data[0][1]
        if sample_flag == 0:
            acc_fft_data = fft(acc_data)
        else:
            acc_fft_data = []
            for sub_acc_data in acc_data:
                acc_fft_data.append(fft(sub_acc_data))
        acc_fft_data = np.array(acc_fft_data)
        emg_data = np.array(emg_data)
        fft_data_list.append([[acc_fft_data, emg_data], data[1], data[2]])
    return fft_data_list


def data_process(data_file_name, sample_flag):
    raw_data_list = json.load(open(data_file_name))
    data_list = prune_data(raw_data_list, sample_flag)
    data_list = fft_process(data_list, sample_flag)
    return np.array(data_list)


def export(operated_data, output_dir):
    # for data in operated_data:
    #     print(len(data[0]))
    print(operated_data.shape, operated_data[0][0][0].shape, operated_data[0][0][1].shape)
    np.save(output_dir, operated_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--output_dir', type=str, default="./emg_fft_data")
    parser.add_argument('--sample_flag', type=int, default=0)
    args = parser.parse_args()
    data_file_name = args.data
    output_dir = args.output_dir
    sample_flag = args.sample_flag
    operated_data = data_process(data_file_name, sample_flag)
    export(operated_data, output_dir)


main()
