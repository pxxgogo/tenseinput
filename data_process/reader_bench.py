import os
import re
import subprocess
import time

dir = "../../data/acc"
start_time = time.time()
filenames = os.listdir(dir)
for filename in filenames:
    if filename.endswith(".txt"):
        output_name = re.sub("_raw", "", filename)
        input_dir = os.path.join(dir, filename)
        subprocess.check_call(["python", "cal_acce.py", input_dir, output_name])
    print("Finish ", output_name, time.time() - start_time)
