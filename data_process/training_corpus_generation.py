import json
import argparse
import numpy as np
import random
import os
import re

RELAX_NUM_PER_USER = 290
RELAX_NUM_FREE_PER_USER = 290
EXTRA_NUM_PER_USER = 240
SEED_0 = 1136
SEED_1 = 514
SEED_2 = 1108
SEED_3 = 923


def generate_data(tag, file_dir, user_id):
    raw_data = np.load(file_dir)
    if tag == 1:
        tense_list = []
        relax_list = []
        for i, data in enumerate(raw_data):
            if data[1] == 1:
                tense_list.append([data[0], 1, data[2], user_id, i])
            else:
                relax_list.append([data[0], 0, data[2], user_id, i])
        random.seed(SEED_0)
        random.shuffle(relax_list)
        relax_list = relax_list[:RELAX_NUM_PER_USER]
        tense_list.extend(relax_list)
        return np.array(tense_list)
    elif tag == 0:
        freestyle_list = []
        for i, data in enumerate(raw_data):
            freestyle_list.append([data[0], 0, data[2], user_id, i])
        random.seed(SEED_1)
        random.shuffle(freestyle_list)
        return np.array(freestyle_list[:RELAX_NUM_FREE_PER_USER])
    else:
        extra_list = []
        for i, data in enumerate(raw_data):
            extra_list.append([data[0], 0, data[2], user_id, i])
        random.seed(SEED_2)
        random.shuffle(extra_list)
        return np.array(extra_list[:EXTRA_NUM_PER_USER])


def read_data(input_dir, users_info):
    EXTRA_COMPILER = re.compile('extra')
    T_R_COMPILER = re.compile('t_r')
    RELAX_COMPILER = re.compile('relax')
    filenames = os.listdir(input_dir)
    total_data = []
    for filename in filenames:
        if filename.endswith(".npy"):
            extra_ok = EXTRA_COMPILER.findall(filename)
            t_r_ok = T_R_COMPILER.findall(filename)
            relax_ok = RELAX_COMPILER.findall(filename)
            file_path = os.path.join(input_dir, filename)
            if len(extra_ok) > 0:
                user_name = re.sub("_extra.npy", "", filename)
                file_tag = 2
            elif len(t_r_ok) > 0:
                user_name = re.sub("_t_r.npy", "", filename)
                file_tag = 1
            else:
                user_name = re.sub("_relax.npy", "", filename)
                file_tag = 0
            user_id = users_info[user_name]
            sub_data = generate_data(file_tag, file_path, user_id)
            total_data.append(sub_data)
    total_data = np.concatenate(total_data)
    np.random.seed(SEED_3)
    np.random.shuffle(total_data)
    training_num = len(total_data) // 10 * 9
    test_data = np.array(total_data[training_num:])
    training_data = np.array(total_data[:training_num])
    return training_data, test_data


def export(training_data, test_data, output_dir):
    np.save(os.path.join(output_dir, "training_data"), training_data)
    np.save(os.path.join(output_dir, "test_data"), test_data)
    di = {}
    di["training_data"] = "training_data.npy"
    di["test_data"] = "test_data.npy"
    config_handle = open(os.path.join(output_dir, "corpus_config.json"), 'w')
    json.dump(di, config_handle)
    config_handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--output_dir', type=str, default="../data/training_data")
    parser.add_argument('--users', type=str, default="./users.json")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    users_path = args.users
    users_info = json.load(open(users_path))
    training_data, test_data = read_data(input_dir, users_info)
    export(training_data, test_data, output_dir)


if __name__ == "__main__":
    main()
