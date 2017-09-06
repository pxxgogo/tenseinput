import argparse
import csv
import json

IGNORED_PADDING_TIME = 200
IGNORED_PADDING_TIME_END = 200
IGNORED_PADDING_TIME_TOTAL = 10000
GENERATION_INTERVAL_TIME = 1500


def reader_csv(csv_file_name):
    key_timestamps = []
    with open(csv_file_name, 'r') as csv_file_handle:
        spamreader = csv.reader(csv_file_handle)
        title_flag = False
        for row in spamreader:
            if not title_flag:
                title_flag = True
                continue
            try:
                gesture_no = int(row[0])
                tense_flag = int(row[1])
                start_time = int(row[2])
                end_time = int(row[3])
            except Exception as e:
                print("first line wrong", csv_file_name)
                continue
            if tense_flag == 1:
                tense_flag = 3
            else:
                tense_flag = 1
            key_timestamps.append((start_time, end_time, tense_flag, gesture_no))
    # print(key_timestamps)
    return key_timestamps


def generate_key_timestamps(acc_file_name, emg_file_name):
    with open(acc_file_name) as file_handle:
        acc_raw_data_list = file_handle.readlines()
    with open(emg_file_name) as file_handle:
        emg_raw_data_list = file_handle.readlines()
    acc_start_data = [float(x) for x in acc_raw_data_list[0].split()]
    acc_start_time = acc_start_data[0] + IGNORED_PADDING_TIME_TOTAL
    emg_start_data = [float(x) for x in emg_raw_data_list[0].split()]
    emg_start_time = emg_start_data[0] + IGNORED_PADDING_TIME_TOTAL

    acc_end_data = [float(x) for x in acc_raw_data_list[len(acc_raw_data_list) - 1].split()]
    acc_end_time = acc_end_data[0] - IGNORED_PADDING_TIME_TOTAL
    emg_end_data = [float(x) for x in emg_raw_data_list[len(emg_raw_data_list) - 1].split()]
    emg_end_time = emg_end_data[0] - IGNORED_PADDING_TIME_TOTAL

    if acc_start_time < emg_start_time:
        start_time = emg_start_time
    else:
        start_time = acc_start_time

    if acc_end_time < emg_end_time:
        end_time = acc_end_time
    else:
        end_time = emg_end_time

    key_timestamps = []
    time = start_time

    while time < end_time:
        key_timestamps.append(
            (time, time + GENERATION_INTERVAL_TIME, 0, -1)
        )
        time += GENERATION_INTERVAL_TIME
    return key_timestamps


def reader(acc_emg_file_name, data_type, key_timestamps, tense_tag):
    acc_emg_data_list = []
    if data_type == 'acc':
        with open(acc_emg_file_name, 'r') as acc_emg_file_handle:
            acc_emg_raw_data_list = acc_emg_file_handle.readlines()
        if tense_tag == 1:
            key_No = 0
            gestures_num = len(key_timestamps)
            choose_flag = False
            acc_emg_data_per_gesture = [[], [], []]
            for data_str in acc_emg_raw_data_list:
                acc_emg_raw_data = [float(x) for x in data_str.split()]
                while acc_emg_raw_data[0] >= key_timestamps[key_No][1] - IGNORED_PADDING_TIME_END and not choose_flag:
                    key_No += 1
                if choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][1] - IGNORED_PADDING_TIME_END:
                    acc_emg_data_list.append(
                        [acc_emg_data_per_gesture, key_timestamps[key_No][2], key_timestamps[key_No][3], key_No])
                    acc_emg_data_per_gesture = [[], [], []]
                    choose_flag = False
                    key_No += 1
                    if key_No >= gestures_num:
                        break
                    continue
                if not choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][0] + IGNORED_PADDING_TIME:
                    choose_flag = True
                if choose_flag:
                    acc_emg_data_per_gesture[0].append(acc_emg_raw_data[1])
                    acc_emg_data_per_gesture[1].append(acc_emg_raw_data[2])
                    acc_emg_data_per_gesture[2].append(acc_emg_raw_data[3])
        else:
            key_No = 0
            gestures_num = len(key_timestamps)
            choose_flag = False
            acc_emg_data_per_gesture = [[], [], []]
            for data_str in acc_emg_raw_data_list:
                acc_emg_raw_data = [float(x) for x in data_str.split()]
                while acc_emg_raw_data[0] >= key_timestamps[key_No][1] - IGNORED_PADDING_TIME_END and not choose_flag:
                    key_No += 1
                if choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][1] - IGNORED_PADDING_TIME_END:
                    if tense_tag == 0:
                        acc_emg_data_list.append([acc_emg_data_per_gesture, tense_tag, -1, key_No])
                    else:
                        acc_emg_data_list.append([acc_emg_data_per_gesture, tense_tag, -2, key_No])
                    acc_emg_data_per_gesture = [[], [], []]
                    choose_flag = False
                    key_No += 1
                    if key_No >= gestures_num:
                        break
                    continue
                if not choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][0] + IGNORED_PADDING_TIME:
                    choose_flag = True
                if choose_flag:
                    acc_emg_data_per_gesture[0].append(acc_emg_raw_data[1])
                    acc_emg_data_per_gesture[1].append(acc_emg_raw_data[2])
                    acc_emg_data_per_gesture[2].append(acc_emg_raw_data[3])
    else:
        with open(acc_emg_file_name, 'r') as acc_emg_file_handle:
            acc_emg_raw_data_list = acc_emg_file_handle.readlines()
        if tense_tag == 1:
            key_No = 0
            gestures_num = len(key_timestamps)
            choose_flag = False
            acc_emg_data_per_gesture = [[], [], [], [], [], []]
            for data_str in acc_emg_raw_data_list:
                acc_emg_raw_data = [int(x) for x in data_str.split()]
                while acc_emg_raw_data[0] >= key_timestamps[key_No][1] - IGNORED_PADDING_TIME_END and not choose_flag:
                    key_No += 1
                if choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][1] - IGNORED_PADDING_TIME_END:
                    acc_emg_data_list.append(
                        [acc_emg_data_per_gesture, key_timestamps[key_No][2], key_timestamps[key_No][3], key_No])
                    acc_emg_data_per_gesture = [[], [], [], [], [], []]
                    choose_flag = False
                    key_No += 1
                    if key_No >= gestures_num:
                        break
                    continue
                if not choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][0] + IGNORED_PADDING_TIME:
                    choose_flag = True
                if choose_flag:
                    acc_emg_data_per_gesture[0].append(acc_emg_raw_data[1])
                    acc_emg_data_per_gesture[1].append(acc_emg_raw_data[2])
                    acc_emg_data_per_gesture[2].append(acc_emg_raw_data[3])
                    acc_emg_data_per_gesture[3].append(acc_emg_raw_data[6])
                    acc_emg_data_per_gesture[4].append(acc_emg_raw_data[7])
                    acc_emg_data_per_gesture[5].append(acc_emg_raw_data[8])
        else:
            key_No = 0
            gestures_num = len(key_timestamps)
            choose_flag = False
            acc_emg_data_per_gesture = [[], [], [], [], [], []]
            for data_str in acc_emg_raw_data_list:
                acc_emg_raw_data = [int(x) for x in data_str.split()]
                while acc_emg_raw_data[0] >= key_timestamps[key_No][1] - IGNORED_PADDING_TIME_END and not choose_flag:
                    key_No += 1
                if choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][1] - IGNORED_PADDING_TIME_END:
                    if tense_tag == 0:
                        acc_emg_data_list.append([acc_emg_data_per_gesture, tense_tag, -1, key_No])
                    else:
                        acc_emg_data_list.append([acc_emg_data_per_gesture, tense_tag, -2, key_No])
                    acc_emg_data_per_gesture = [[], [], [], [], [], []]
                    choose_flag = False
                    key_No += 1
                    if key_No >= gestures_num:
                        break
                    continue
                if not choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][0] + IGNORED_PADDING_TIME:
                    choose_flag = True
                if choose_flag:
                    acc_emg_data_per_gesture[0].append(acc_emg_raw_data[1])
                    acc_emg_data_per_gesture[1].append(acc_emg_raw_data[2])
                    acc_emg_data_per_gesture[2].append(acc_emg_raw_data[3])
                    acc_emg_data_per_gesture[3].append(acc_emg_raw_data[6])
                    acc_emg_data_per_gesture[4].append(acc_emg_raw_data[7])
                    acc_emg_data_per_gesture[5].append(acc_emg_raw_data[8])
    return acc_emg_data_list


def merge_data(acc_data_list, emg_data_list):
    acc_index = 0
    emg_index = 0
    data_list = []
    while acc_index < len(acc_data_list) and emg_index < len(emg_data_list):
        acc_data = acc_data_list[acc_index]
        emg_data = emg_data_list[emg_index]
        if acc_data[3] == emg_data[3]:
            data_list.append(
                [[acc_data[0], emg_data[0]], acc_data[1], acc_data[2]]
            )
            acc_index += 1
            emg_index += 1
        elif acc_data[3] > emg_data[3]:
            emg_index += 1
        else:
            acc_index += 1
    return data_list


def process(acc_file_name, emg_file_name, key_timestamps, tense_tag):
    acc_data_list = reader(acc_file_name, 'acc', key_timestamps, tense_tag)
    emg_data_list = reader(emg_file_name, 'emg', key_timestamps, tense_tag)
    data_list = merge_data(acc_data_list, emg_data_list)
    return data_list


def print_data(acc_data_list, output_dir):
    with open(output_dir, 'w') as file_handle:
        file_handle.write(json.dumps(acc_data_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', type=int)
    parser.add_argument('acc', type=str)
    parser.add_argument('emg', type=str)
    parser.add_argument('--csv', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="./acc_data.json")
    args = parser.parse_args()
    csv_file_name = args.csv
    acc_file_name = args.acc
    emg_file_name = args.emg
    tag = args.tag
    output_dir = args.output_dir
    if tag > 0:
        key_timestamps = reader_csv(csv_file_name)
    else:
        key_timestamps = generate_key_timestamps(acc_file_name, emg_file_name)
    acc_emg_data_list = process(acc_file_name, emg_file_name, key_timestamps, tag)
    print_data(acc_emg_data_list, output_dir)


main()
