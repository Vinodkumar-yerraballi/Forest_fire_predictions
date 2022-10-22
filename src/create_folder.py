import os
import shutil
import random
import yaml
import argparse
import numpy as np
import pandas as pd
from get_data import get_data

#################### CREATING FOLDER - STAR ########################

def create_fold(config, img=None):
    config=get_data(config)
    dirr=config['load_data']['preprocessed_data']
    cla=config['load_data']['num_classes']
    print(dirr)
    print(cla)
    if os.path.exists(dirr+'/'+'train'+'class_0') and os.path.exists(dirr+'/'+'test'+'class_0'):
        print("train and test folders are already exists...!")
        print("I am Skipping it...!")
    else:
        os.mkdir(dirr+'/'+'train')
        os.mkdir(dirr+'/'+'test')
        for i in range(cla):
            os.makedirs(os.path.join(dirr+'/'+'train','class_'+str(i)))
            os.makedirs(os.path.join(dirr+'/'+'test','class_'+str(i)))

#################### CREATING FOLDER - END ########################


if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--config", default='params.yaml')
    passed_args=args.parse_args()
    create_fold(config=passed_args.config)