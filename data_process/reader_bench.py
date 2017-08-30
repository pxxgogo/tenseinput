import os
import re
import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_input_dir', type=str)
parser.add_argument('log_input_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('type', type=str)


args = parser.parse_args()
data_input_dir = args.data_input_dir
log_input_dir = args.log_input_dir
output_dir = args.output_dir
file_type = args.type
EXTRA_COMPILER = re.compile('extra')
T_R_COMPILER = re.compile('t_r')
RELAX_COMPILER = re.compile('relax')

start_time = time.time()
filenames = os.listdir(data_input_dir)
for filename in filenames:
    if filename.endswith(".txt"):
        extra_ok = EXTRA_COMPILER.findall(filename)
        t_r_ok = T_R_COMPILER.findall(filename)
        relax_ok = RELAX_COMPILER.findall(filename)
        input_data_path = os.path.join(data_input_dir, filename)
        output_name = re.sub(".txt", ".json", filename)
        output_path = os.path.join(output_dir, output_name)
        if len(extra_ok) > 0:
            log_name = re.sub(".txt", "_log.csv", filename)
            log_path = os.path.join(log_input_dir, log_name)
            subprocess.check_call(["python", "reader.py", '2', input_data_path, '--csv', log_path, '--type', file_type, '--output_dir', output_path])
        elif len(t_r_ok) > 0:
            log_name = re.sub(".txt", "_log.csv", filename)
            log_path = os.path.join(log_input_dir, log_name)
            subprocess.check_call(["python", "reader.py", '1', input_data_path, '--csv', log_path, '--type', file_type, '--output_dir', output_path])
        else:
            subprocess.check_call(["python", "reader.py", '0', input_data_path, '--type', file_type, '--output_dir', output_path])
        print("Finish ", output_name, time.time() - start_time)
