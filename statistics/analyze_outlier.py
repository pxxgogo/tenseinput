import json
import os
import argparse
import re

COST_THRESHOLD = 1.6


def extract_outliers(filename, key_model_No):
    key_model_No += 1
    with open(filename) as filehandle:
        lines = filehandle.readlines()
    start_flag = True
    outliers = []
    for line in lines:
        if start_flag:
            start_flag = False
            continue
        info = line[:-1].split(",")
        if key_model_No != int(info[0]):
            continue
        cost = float(info[5])
        if cost > COST_THRESHOLD:
            outliers.append(info)
    return outliers


def generate_outliers_config(total_outliers):
    config_info = []
    config_data = []
    print(len(total_outliers))
    gesture_types_outliers = {}
    for outlier in total_outliers:
        config_info.append([int(outlier[2]), int(outlier[3]), int(outlier[4]), float(outlier[5])])
        gesture_types_outliers[int(outlier[2])] = gesture_types_outliers.get(int(outlier[2]), 0) + 1
        config_data.append(",".join(outlier[2:5]))
    for key, value in gesture_types_outliers.items():
        print(key, value)
    return config_data, config_info


parser = argparse.ArgumentParser()
parser.add_argument('statistics_dir', type=str)
parser.add_argument('--config_name', type=str, default="config.json")
parser.add_argument('output_dir', type=str)
parser.add_argument('--output_name', type=str, default='outliers')

args = parser.parse_args()
statistics_dir = args.statistics_dir
config_name = args.config_name
output_dir = args.output_dir
output_name = args.output_name
filenames = os.listdir(statistics_dir)
total_outliers = []
config_path = os.path.join(statistics_dir, config_name)
config = json.load(open(config_path))
for filename in filenames:
    if filename.endswith(".csv"):
        file_path = os.path.join(statistics_dir, filename)
        key_word = re.sub(".csv", "", filename)
        outliers = extract_outliers(file_path, config[key_word])
        total_outliers.extend(outliers)
config_data, config_info = generate_outliers_config(total_outliers)
config_data_path = os.path.join(output_dir, "%s_data.json" % output_name)
config_info_path = os.path.join(output_dir, "%s_info.json" % output_name)

with open(config_data_path, 'w') as file_handle:
    file_handle.write(json.dumps(config_data))

with open(config_info_path, 'w') as file_handle:
    file_handle.write(json.dumps(config_info))
