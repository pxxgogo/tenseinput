import os
import re
import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--estimator', type=str, default="1")


args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
estimator_flag = args.estimator
start_time = time.time()
filenames = os.listdir(input_dir)
for filename in filenames:
    if filename.endswith(".txt"):
        output_name = re.sub("_raw", "", filename)
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, output_name)
        subprocess.check_call(["python", "cal_acce.py", input_path, output_path, "--estimator", estimator_flag])
    print("Finish ", output_name, time.time() - start_time)
