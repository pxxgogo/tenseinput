import os
import re
import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('acc_input_dir', type=str)
parser.add_argument('emg_input_dir', type=str)
parser.add_argument('log_input_dir', type=str)
parser.add_argument('output_dir', type=str)

args = parser.parse_args()
acc_input_dir = args.acc_input_dir
emg_input_dir = args.emg_input_dir
log_input_dir = args.log_input_dir
output_dir = args.output_dir
EXTRA_COMPILER = re.compile('extra')
T_R_COMPILER = re.compile('t_r')
RELAX_COMPILER = re.compile('relax')

start_time = time.time()
filenames = os.listdir(acc_input_dir)
for filename in filenames:
    if filename.endswith(".txt"):
        extra_ok = EXTRA_COMPILER.findall(filename)
        t_r_ok = T_R_COMPILER.findall(filename)
        relax_ok = RELAX_COMPILER.findall(filename)
        acc_data_path = os.path.join(acc_input_dir, filename)
        emg_data_path = os.path.join(emg_input_dir, filename)
        if not os.access(acc_data_path, os.R_OK) or not os.access(emg_data_path, os.R_OK):
            print("error", filename)
            continue
        output_name = re.sub(".txt", ".json", filename)
        output_path = os.path.join(output_dir, output_name)
        if len(extra_ok) > 0:
            log_name = re.sub(".txt", "_log.csv", filename)
            log_path = os.path.join(log_input_dir, log_name)
            subprocess.check_call(
                ["python", "mix_reader.py", '2', acc_data_path, emg_data_path, '--csv', log_path, '--output_dir',
                 output_path])
        elif len(t_r_ok) > 0:
            log_name = re.sub(".txt", "_log.csv", filename)
            log_path = os.path.join(log_input_dir, log_name)
            subprocess.check_call(
                ["python", "mix_reader.py", '1', acc_data_path, emg_data_path, '--csv', log_path, '--output_dir',
                 output_path])
        else:
            subprocess.check_call(
                ["python", "mix_reader.py", '0', acc_data_path, emg_data_path, '--output_dir', output_path])
        print("Finish ", output_name, time.time() - start_time)
