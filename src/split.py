from logging import root
import os
import shutil
import random
import yaml
import argparse
import numpy as np
import pandas as pd
from get_data import get_data

def train_and_test(config_file):
    config = get_data(config_file)
    root_dir = config['data_source']['data_src']
    dest = config['load_data']['preprocessed_data']
    p = config['load_data']['full_Path']
    cla = config['data_source']['data_src']
    cla = os.listdir(cla)
    
    splitr = config['train_split']['split_ratio']
    for k in range(len(cla)):
        per = len(os.listdir((os.path.join(root_dir,cla[k]))))
        print(k,"->",per)
        cnt = 0
        split_ratio = round((splitr/100)*per)
        for j in os.listdir((os.path.join(root_dir,cla[k]))):
            pat = os.path.join(root_dir+'/'+cla[k],j)
            # print(pat)
            if(cnt!=split_ratio):
                shutil.copy(pat, dest+'/'+'train/class_'+str(k))
                cnt+=1
            else:
                shutil.copy(pat, dest+'/'+'test/class_'+str(k))
        print('Done')


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args  = args.parse_args()
    train_and_test(config_file=parsed_args.config)