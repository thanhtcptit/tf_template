# -*- coding: utf-8 -*-
import os
import logging
import datetime
import subprocess


def get_current_time_str():
    now = datetime.datetime.now()
    return now.isoformat()


def multi_makedirs(dirs, exist_ok=False):
    if not isinstance(dirs, list):
        dirs = list(dirs)
    for d in dirs:
        os.makedirs(d, exist_ok=exist_ok)


def get_file_logger(file_path):
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(file_path),
            logging.StreamHandler(sys.stdout)
        ])
    return logging.getLogger()


def run_command(command):
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    return p.communicate()
