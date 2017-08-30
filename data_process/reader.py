import argparse
import csv
import json
IGNORED_PADDING_TIME = 250
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


def reader_acc_emg(acc_emg_file_name, key_timestamps, tense_tag, file_type):
    acc_emg_data_list = []
    if file_type == 0:
        with open(acc_emg_file_name, 'r') as acc_emg_file_handle:
            acc_emg_raw_data_list = acc_emg_file_handle.readlines()
            if tense_tag == 1:
                key_No = 0
                gestures_num = len(key_timestamps)
                choose_flag = False
                acc_emg_data_per_gesture = [[], [], []]
                for data_str in acc_emg_raw_data_list:
                    acc_emg_raw_data = [float(x) for x in data_str.split()]
                    if choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][1]:
                        acc_emg_data_list.append([acc_emg_data_per_gesture, key_timestamps[key_No][2]])
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
            elif tense_tag == 2:
                key_No = 0
                gestures_num = len(key_timestamps)
                choose_flag = False
                acc_emg_data_per_gesture = [[], [], []]
                for data_str in acc_emg_raw_data_list:
                    acc_emg_raw_data = [float(x) for x in data_str.split()]
                    if choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][1]:
                        acc_emg_data_list.append([acc_emg_data_per_gesture, tense_tag])
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
                acc_emg_data_per_gesture = [[], [], []]
                for data_str in acc_emg_raw_data_list:
                    acc_emg_raw_data = [float(x) for x in data_str.split()]
                    acc_emg_data_per_gesture[0].append(acc_emg_raw_data[1])
                    acc_emg_data_per_gesture[1].append(acc_emg_raw_data[2])
                    acc_emg_data_per_gesture[2].append(acc_emg_raw_data[3])
                acc_emg_data_list = [acc_emg_data_per_gesture, tense_tag]
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
                    if choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][1]:
                        acc_emg_data_list.append([acc_emg_data_per_gesture, key_timestamps[key_No][2]])
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
            elif tense_tag == 2:
                key_No = 0
                gestures_num = len(key_timestamps)
                choose_flag = False
                acc_emg_data_per_gesture = [[], [], [], [], [], []]
                for data_str in acc_emg_raw_data_list:
                    acc_emg_raw_data = [int(x) for x in data_str.split()]
                    if choose_flag and acc_emg_raw_data[0] >= key_timestamps[key_No][1]:
                        acc_emg_data_list.append([acc_emg_data_per_gesture, tense_tag])
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
                acc_emg_data_per_gesture = [[], [], [], [], [], []]
                for data_str in acc_emg_raw_data_list:
                    acc_emg_raw_data = [int(x) for x in data_str.split()]
                    acc_emg_data_per_gesture[0].append(acc_emg_raw_data[1])
                    acc_emg_data_per_gesture[1].append(acc_emg_raw_data[2])
                    acc_emg_data_per_gesture[2].append(acc_emg_raw_data[3])
                    acc_emg_data_per_gesture[3].append(acc_emg_raw_data[6])
                    acc_emg_data_per_gesture[4].append(acc_emg_raw_data[7])
                    acc_emg_data_per_gesture[5].append(acc_emg_raw_data[8])
                acc_emg_data_list = [acc_emg_data_per_gesture, tense_tag]
    return acc_emg_data_list


def print_data(acc_data_list, output_dir):
    with open(output_dir, 'w') as file_handle:
        file_handle.write(json.dumps(acc_data_list))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag', type=int)
    parser.add_argument('acc_emg', type=str)
    parser.add_argument('--csv', type=str, default="")
    parser.add_argument('--type', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default="./acc_data.json")
    args = parser.parse_args()
    csv_file_name = args.csv
    acc_emg_file_name = args.acc_emg
    tag = args.tag
    output_dir = args.output_dir
    file_type = args.type
    if tag > 0:
        key_timestamps = reader_csv(csv_file_name)
    else:
        key_timestamps = []
    acc_emg_data_list = reader_acc_emg(acc_emg_file_name, key_timestamps, tag, file_type)
    print_data(acc_emg_data_list, output_dir)

main()
