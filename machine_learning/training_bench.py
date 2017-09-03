import os
import re
import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('config_dir', type=str)
parser.add_argument('output_dir', type=str)

args = parser.parse_args()
config_dir = args.config_dir
output_dir = args.output_dir

start_time = time.time()
filenames = os.listdir(config_dir)
for filename in filenames:
    if filename.endswith(".json"):
        config_path = os.path.join(config_dir, filename)
        if not os.access(config_path, os.R_OK):
            print("error", filename)
            continue
        output_name = re.sub(".json", "", filename)
        output_path = os.path.join(output_dir, output_name)
        subprocess.check_call(
            ["python", "kernel.py", '--model_dir', output_path, '--config_dir', config_path])
        print("Finish ", output_name, time.time() - start_time)
