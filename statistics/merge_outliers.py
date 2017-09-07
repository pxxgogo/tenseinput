import json
import os
import argparse
import re


parser = argparse.ArgumentParser()
parser.add_argument('outliers_dir', type=str)
parser.add_argument('output_dir', type=str)

args = parser.parse_args()
outliers_dir = args.outliers_dir
output_dir = args.output_dir
filenames = os.listdir(outliers_dir)
total_outliers = {}
file_num = 0
for filename in filenames:
    if filename.endswith(".json"):
        file_path = os.path.join(outliers_dir, filename)
        outliers = json.load(open(file_path))
        file_num += 1
        for outlier in outliers:
            total_outliers[outlier] = total_outliers.get(outlier, 0) + 1
ret_list = []
for key, value in total_outliers.items():
    if value == file_num:
        ret_list.append(key)
print(len(ret_list))
with open(output_dir, 'w') as file_handle:
    file_handle.write(json.dumps(ret_list))
