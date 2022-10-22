from distutils.command.config import config
import os
import shutil
import random
import yaml
import argparse
import numpy as np
import pandas as pd


def get_data(config_file):
    config=read_params(config_file)
    return config

def read_params(config_file):
    with open(config_file) as conf:
        config=yaml.safe_load(conf)
    return config

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml')
    passed_args=args.parse_args()
    a = get_data(config_file=passed_args.config)