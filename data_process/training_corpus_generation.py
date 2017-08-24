import json
import argparse
import numpy as np
import random
import os

def generate_data(tag, file_dir, input_size, output_size):
    raw_data = np.load(file_dir)
    new_data = []
    tag_data = [0 for i in range(output_size)]
    tag_data[tag] = 1
    for data in raw_data:
        op_data = data[:, :, :input_size // 3]
        op_data = np.reshape(op_data, (op_data.shape[0], -1))
        new_data.append([op_data, tag_data])
    return new_data


def read_data(input_dir, config_name, input_size, output_size):
    config_dir = os.path.join(input_dir, config_name)
    data_config = json.load(open(config_dir, 'r'))
    data = []
    # print(data_config)
    for data_setting in data_config:
        sub_data = generate_data(data_setting['tag'], os.path.join(input_dir, data_setting['file_name']), input_size,
                                 output_size)
        data.extend(sub_data)
    random.shuffle(data)
    training_num = len(data) // 10 * 9
    test_data = np.array(data[training_num:])
    training_data = np.array(data[:training_num])
    return training_data, test_data

def export(training_data, test_data, output_dir, input_size, output_size):
    # print(acc_fft_data.shape)
    np.save(os.path.join(output_dir, "training_data"), training_data)
    np.save(os.path.join(output_dir, "test_data"), test_data)
    # acc_fft_data.(output_dir, " ")
    di = {}
    di["input_size"] = input_size
    di["output_size"] = output_size
    di["training_data"] = "training_data.npy"
    di["test_data"] = "test_data.npy"
    config_handle = open(os.path.join(output_dir, "corpus_config.json"), 'w')
    json.dump(di, config_handle)
    config_handle.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--config_name', type=str, default="mix_config.json")
    parser.add_argument('--output_dir', type=str, default="../data/training_data")
    parser.add_argument('--input_size', type=int, default=600)
    parser.add_argument('--output_size', type=int, default=2)
    args = parser.parse_args()
    input_dir = args.input_dir
    config_name = args.config_name
    config_dir = os.path.join(input_dir, config_name)
    output_dir = args.output_dir
    input_size = args.input_size
    output_size = args.output_size
    training_data, test_data = read_data(input_dir, config_name, input_size, output_size)
    export(training_data, test_data, output_dir, input_size, output_size)


if __name__ == "__main__":
    main()
