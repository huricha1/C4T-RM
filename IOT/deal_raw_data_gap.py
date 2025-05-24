import argparse
import numpy as np
import time
import random
import os
import sklearn.model_selection as skl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
import statistics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
parser = argparse.ArgumentParser()

# experimentName is the current file name without extension

parser.add_argument("--expPATH", type=str, default=os.path.expanduser('C4T-RM/IOT/iot/'),
                    help="Experiment path")
opt = parser.parse_args()
df = pd.read_csv('46.246.fe1d.9416_00100_20180423080830-conn.logreplaced.csv')
df=df.drop(df.iloc[:, 4:7], axis=1)
df=df.drop(df.iloc[:, 6:], axis=1)
# Train/test split
df=df[0:4410].copy()
train_df=df[0:2610].copy()
val_df=df[2610:3510].copy()
test_df=df[3510:].copy()

random.seed(123)

sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
val_df = sc.transform(val_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
val_df = val_df.astype(np.float32)
test_df = test_df.astype(np.float32)

np.save(os.path.join(opt.expPATH, 'iot_gap60_val_ground_truth.npy'), val_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'iot_gap60_test_ground_truth.npy'), test_df, allow_pickle=False)

x=0
for i in range(26):
    idx=random.randint(10,60)
    idx=idx+x
    train_df[idx:idx+60]=np.nan
    x=idx+60
x=0
for i in range(9):
    idx=random.randint(10,60)
    idx=idx+x
    val_df[idx:idx+60]=np.nan
    x=idx+60
x=0
for i in range(9):
    idx=random.randint(10,60)
    idx=idx+x
    test_df[idx:idx+60]=np.nan
    x=idx+60

np.save(os.path.join(opt.expPATH,'iot_gap60_train.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'iot_gap60_test.npy'), test_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'iot_gap60_val.npy'), val_df, allow_pickle=False)
