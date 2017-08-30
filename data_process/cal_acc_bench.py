import os
import re
import subprocess
import time

input_dir = "/home/pxxgogo/tenseinput_data/0828/acc"
output_dir = "/home/pxxgogo/tenseinput/data/raw_data/0828/acc"
start_time = time.time()
filenames = os.listdir(input_dir)
for filename in filenames:
    if filename.endswith(".txt"):
        output_name = re.sub("_raw", "", filename)
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, output_name)
        subprocess.check_call(["python", "cal_acce.py", input_path, output_path])
    print("Finish ", output_name, time.time() - start_time)
