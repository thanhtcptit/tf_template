# -*- coding: utf-8 -*-
import os
import re
import time
import shutil
import subprocess


def main(config_dir, log_dir, model_type, recover=True, force=False,
         display_logs=True):
    os.makedirs(log_dir, exist_ok=True)

    config_files = sorted(os.listdir(config_dir))
    for config_file in config_files:
        config_file_path = os.path.join(config_dir, config_file)
        exp_name, ext = os.path.splitext(config_file)
        if ext != '.json':
            continue
        prefix = re.findall('^P[0-9]+_', exp_name)
        if prefix:
            exp_name = exp_name[exp_name.index(prefix[0]) + len(prefix):]
        exp_train_log = os.path.join(log_dir, exp_name)
        if os.path.exists(exp_train_log) and not force and not recover:
            print(f'***** Skip exp {config_file}: Exp dir {exp_train_log} '
                  'already exists and no action is specified *****')
            continue
        os.makedirs(exp_train_log, exist_ok=True)

        run_command = (f'python3 run.py train {config_file_path} '
                       f'-s {exp_train_log} -t {model_type}')
        if recover:
            run_command += ' -r'
        if force:
            run_command += ' -f'

        print(f'***** START EXPERIMENT: {config_file} ******')
        print('Run command: ', run_command)
        start_time = time.time()
        p = subprocess.Popen(
            run_command,
            stdout=None if display_logs else subprocess.DEVNULL,
            stderr=None if display_logs else subprocess.DEVNULL,
            shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        print(f'***** FINISHED. Status: {p_status} - '
              f'Time: {time.time() - start_time} *****')
