import os
import re
import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--fft', type=str, default='0')
parser.add_argument('--data_format', type=str, default='0')

args = parser.parse_args()
data_input_dir = args.data_input_dir
output_dir = args.output_dir
fft_flag = args.fft
data_format = args.data_format

start_time = time.time()
filenames = os.listdir(data_input_dir)
for filename in filenames:
    if filename.endswith(".json"):
        input_data_path = os.path.join(data_input_dir, filename)
        output_name = re.sub(".json", "", filename)
        output_path = os.path.join(output_dir, output_name)

        subprocess.check_call(
            ["python", "emg_operation.py", input_data_path, '--fft', fft_flag, '--output_dir', output_path,
             '--data_format', data_format])

        print("Finish ", output_name, time.time() - start_time)
