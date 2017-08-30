import os
import re
import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--fft', type=str, default='0')

args = parser.parse_args()
data_input_dir = args.data_input_dir
output_dir = args.output_dir
fft_flag = args.fft
EXTRA_COMPILER = re.compile('extra')
T_R_COMPILER = re.compile('t_r')
RELAX_COMPILER = re.compile('relax')

start_time = time.time()
filenames = os.listdir(data_input_dir)
for filename in filenames:
    if filename.endswith(".json"):
        extra_ok = EXTRA_COMPILER.findall(filename)
        t_r_ok = T_R_COMPILER.findall(filename)
        relax_ok = RELAX_COMPILER.findall(filename)
        input_data_path = os.path.join(data_input_dir, filename)
        output_name = re.sub(".json", "", filename)
        output_path = os.path.join(output_dir, output_name)
        if len(extra_ok) > 0:
            tag_type = '2'

        elif len(t_r_ok) > 0:
            tag_type = '1'
        else:
            tag_type = '0'
        subprocess.check_call(
            ["python", "emg_operation.py", tag_type, input_data_path, '--fft', fft_flag, '--output_dir', output_path])

        print("Finish ", output_name, time.time() - start_time)
