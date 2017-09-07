import json
import argparse
import numpy as np
import random
import os
import re
import subprocess

RELAX_NUM_PER_USER = 240
RELAX_NUM_FREE_PER_USER = 240
EXTRA_NUM_PER_USER = 240
SEED_0 = 1136
SEED_1 = 514
SEED_2 = 1108
SEED_3 = 923
SEED_4 = 722
OUTPUT_DIR = "./tmp_corpus"


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
        return tense_list
    elif tag == 0:
        freestyle_list = []
        for i, data in enumerate(raw_data):
            freestyle_list.append([data[0], 0, data[2], user_id, i])
        random.seed(SEED_1)
        random.shuffle(freestyle_list)
        return freestyle_list[:RELAX_NUM_FREE_PER_USER]
    elif tag == 3:
        e_t_list = []
        for i, data in enumerate(raw_data):
            e_t_list.append([data[0], 1, -3, user_id, i])
        random.seed(SEED_4)
        random.shuffle(e_t_list)
        return e_t_list[:EXTRA_NUM_PER_USER]
    else:
        extra_list = []
        for i, data in enumerate(raw_data):
            extra_list.append([data[0], 0, data[2], user_id, i])
        random.seed(SEED_2)
        random.shuffle(extra_list)
        return extra_list[:EXTRA_NUM_PER_USER]


def split_corpus(total_data):
    unit_length = len(total_data) // 10
    print(unit_length)
    corpus_parts = []
    for i in range(10):
        corpus_parts.append(total_data[i * unit_length: (i + 1) * unit_length])
    return corpus_parts


def export(training_data, test_data, output_dir):
    print("training_data:", training_data.shape, training_data[0][0][0].shape, training_data[0][0][1].shape)
    print("test_data:", test_data.shape, test_data[0][0][0].shape, test_data[0][0][1].shape)
    np.save(os.path.join(output_dir, "training_data"), training_data)
    np.save(os.path.join(output_dir, "test_data"), test_data)
    di = {}
    di["training_data"] = "training_data.npy"
    di["test_data"] = "test_data.npy"
    config_handle = open(os.path.join(output_dir, "corpus_config.json"), 'w')
    json.dump(di, config_handle)
    config_handle.close()


def generate_training_corpus(corpus_parts, i):
    training_corpus = []
    test_corpus = np.array(corpus_parts[i])
    for j in range(len(corpus_parts)):
        if j != i:
            training_corpus.append(corpus_parts[j])

    return np.concatenate(training_corpus), test_corpus

def train(corpus_parts):
    for i in range(len(corpus_parts)):
        training_corpus, test_corpus = generate_training_corpus(corpus_parts, i)
        export(training_corpus, test_corpus, OUTPUT_DIR)
        subprocess.check_call(["python", "mix_kernel.py", "--model_dir", "./models/model_%d" % i])



def process(input_dir, users_info):
    EXTRA_COMPILER = re.compile('extra')
    T_R_COMPILER = re.compile('t_r')
    RELAX_COMPILER = re.compile('relax')
    E_T_COMPILER = re.compile('e_t')

    filenames = os.listdir(input_dir)
    total_data = []
    for filename in filenames:
        if filename.endswith(".npy"):
            extra_ok = EXTRA_COMPILER.findall(filename)
            t_r_ok = T_R_COMPILER.findall(filename)
            relax_ok = RELAX_COMPILER.findall(filename)
            e_t_ok = E_T_COMPILER.findall(filename)
            file_path = os.path.join(input_dir, filename)
            if len(extra_ok) > 0:
                user_name = re.sub("_extra.npy", "", filename)
                file_tag = 2
            elif len(t_r_ok) > 0:
                user_name = re.sub("_t_r.npy", "", filename)
                file_tag = 1
            elif len(e_t_ok) > 0:
                user_name = re.sub("_e_t.npy", "", filename)
                file_tag = 3
            else:
                user_name = re.sub("_relax.npy", "", filename)
                file_tag = 0
            user_id = users_info[user_name]
            sub_data = generate_data(file_tag, file_path, user_id)
            total_data.append(sub_data)
    total_data = np.concatenate(total_data)
    np.random.seed(SEED_3)
    np.random.shuffle(total_data)
    corpus_parts = split_corpus(total_data)
    train(corpus_parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--users', type=str, default="./users.json")
    args = parser.parse_args()
    input_dir = args.input_dir
    users_path = args.users
    users_info = json.load(open(users_path))
    process(input_dir, users_info)


if __name__ == "__main__":
    main()
