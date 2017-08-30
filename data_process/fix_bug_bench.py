import os
import re
import subprocess
import time

dir = "/home/pxxgogo/tenseinput_data/0822/acc/raw_with_bug"
output_dir = "/home/pxxgogo/tenseinput_data/0822/acc"
start_time = time.time()
filenames = os.listdir(dir)
for filename in filenames:
    if filename.endswith(".txt"):
        input_dir = os.path.join(dir, filename)
        output_name = os.path.join(output_dir, filename)
        subprocess.check_call(["python", "fix_bug.py", input_dir, output_name])
    print("Finish ", filename, time.time() - start_time)
