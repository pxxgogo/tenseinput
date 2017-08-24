import argparse
import os

from estimator import Estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_name', type=str)
    parser.add_argument('--output_dir', type=str, default="../data/raw_data/0822")
    args = parser.parse_args()
    input_path = args.input_path
    output_dir = args.output_dir
    output_name = args.output_name
    output_path = os.path.join(output_dir, output_name)
    output_file_handle = open(output_path, 'w')
    est = Estimator()
    with open(input_path) as file_handle:
        raw_data_list = file_handle.readlines()
    last_timestamp = -1
    for raw_data_str in raw_data_list:
        raw_data_str = raw_data_str[:-1]
        raw_data = [x for x in raw_data_str.split()]
        timestamp = float(raw_data[0])
        acce_raw_data = [int(x) for x in raw_data[4:7]]
        gyro_data = [int(x) for x in raw_data[1:4]]
        if last_timestamp == -1:
            last_timestamp = timestamp
            continue
        acce_data = est.feed_data(timestamp - last_timestamp, gyro_data, acce_raw_data)
        new_acce_data = acce_data * 4096
        ts = round(timestamp * 1000)
        # new_acce_data = [float(x) for x in new_acce_data]
        # print(timestamp - last_timestamp, acce_data, new_acce_data)
        output_file_handle.write(str(ts) + ' ' + str(new_acce_data[0]) + ' ' + str(new_acce_data[1]) + ' ' + str(new_acce_data[2]) + '\n')
        last_timestamp = timestamp
    output_file_handle.close()
