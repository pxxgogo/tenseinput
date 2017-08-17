import argparse
import csv
import json

def reader_csv(csv_file_name):
    key_timestamps = []
    with open(csv_file_name, 'r') as csv_file_handle:
        spamreader = csv.reader(csv_file_handle)
        title_flag = False
        for row in spamreader:
            if not title_flag:
                title_flag = True
                continue
            start_time = int(row[2])
            end_time = int(row[3])
            key_timestamps.append((start_time, end_time))
    # print(key_timestamps)
    return key_timestamps


def reader_acc(acc_file_name, key_timestamps, tense_tag):
    acc_data_list = []
    with open(acc_file_name, 'r') as acc_file_handle:
        acc_raw_data_list = acc_file_handle.readlines()
        acc_raw_data_length = len(acc_raw_data_list)
        if tense_tag > 0:
            key_No = 0
            gestures_num = len(key_timestamps)
            choose_flag = False
            acc_data_per_gesture = [[], [], []]
            for data_str in acc_raw_data_list:
                acc_raw_data = [float(x) for x in data_str.split()]
                if choose_flag and acc_raw_data[0] >= key_timestamps[key_No][1]:
                    choose_flag = False
                    key_No += 1
                    acc_data_list.append([acc_data_per_gesture, tense_tag])
                    acc_data_per_gesture = [[], [], []]
                    if key_No >= gestures_num:
                        break
                    continue
                if not choose_flag and acc_raw_data[0] >= key_timestamps[key_No][0]:
                    choose_flag = True
                if choose_flag:
                    acc_data_per_gesture[0].append(acc_raw_data[1])
                    acc_data_per_gesture[1].append(acc_raw_data[2])
                    acc_data_per_gesture[2].append(acc_raw_data[3])
        else:
            acc_data_per_gesture = [[], [], []]
            for data_str in acc_raw_data_list:
                acc_raw_data = [float(x) for x in data_str.split()]
                acc_data_per_gesture[0].append(acc_raw_data[1])
                acc_data_per_gesture[1].append(acc_raw_data[2])
                acc_data_per_gesture[2].append(acc_raw_data[3])
            acc_data_list = [acc_data_per_gesture, tense_tag]

    return acc_data_list


def print_data(acc_data_list, output_dir):
    with open(output_dir, 'w') as file_handle:
        file_handle.write(json.dumps(acc_data_list))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', type=int)
    parser.add_argument('acc', type=str)
    parser.add_argument('--csv', type=str)
    parser.add_argument('--output_dir', type=str, default="./acc_data.json")
    args = parser.parse_args()
    csv_file_name = args.csv
    acc_file_name = args.acc
    tag = args.tag
    output_dir = args.output_dir
    if tag > 0:
        key_timestamps = reader_csv(csv_file_name)
    else:
        key_timestamps = []
    acc_data_list = reader_acc(acc_file_name, key_timestamps, tag)
    print_data(acc_data_list, output_dir)

main()
