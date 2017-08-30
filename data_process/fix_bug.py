import argparse
import json

def reader_acc(acc_file_name):
    acc_data_list = []
    with open(acc_file_name, 'r') as acc_file_handle:
        acc_raw_data_list = acc_file_handle.readlines()
        acc_data_list = []
        for data_str in acc_raw_data_list:
            acc_raw_data = [x for x in data_str.split()]
            acc_data = [acc_raw_data[0], acc_raw_data[6], acc_raw_data[1], acc_raw_data[2], acc_raw_data[3], acc_raw_data[4], acc_raw_data[5]]
            acc_data_list.append(" ".join(acc_data) + "\n")
    return acc_data_list


def print_data(acc_data_list, output_dir):
    with open(output_dir, 'w') as file_handle:
        file_handle.writelines(acc_data_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('acc', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    acc_file_name = args.acc
    output_dir = args.output_dir
    acc_data_list = reader_acc(acc_file_name)
    print_data(acc_data_list, output_dir)


main()
