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
experimentName = os.path.splitext(os.path.basename(__file__))[0]
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('C4T-RM/BATADAL/batadal/'),
                    help="Experiment path")

opt = parser.parse_args()
df = pd.read_csv('BATADAL_dataset03.csv')


df=df.drop(['DATETIME'], axis=1)
df=df.drop(['ATT_FLAG'], axis=1)
df=df.drop(['S_PU11'], axis=1)
df=df.drop(['F_PU11'], axis=1)
df=df.drop(['S_PU9'], axis=1)
df=df.drop(['F_PU9'], axis=1)
df=df.drop(['S_PU3'], axis=1)
df=df.drop(['F_PU3'], axis=1)
df=df.drop(['S_PU1'], axis=1)
# Train/test split
df=df[0:8760].copy()
train_df=df[0:5160].copy()
val_df=df[5160:6960].copy()
test_df=df[6960:].copy()
print(test_df.shape,val_df.shape)
random.seed(123)
sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
val_df = sc.transform(val_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
val_df = val_df.astype(np.float32)
test_df = test_df.astype(np.float32)

np.save(os.path.join(opt.expPATH, 'batadal_gap60_val_ground_truth.npy'), val_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'batadal_gap60_test_ground_truth.npy'), test_df, allow_pickle=False)
print(np.min(test_df),np.max(test_df))
print(np.min(val_df),np.max(val_df))
# print((test_df==-1.0).any())
# print((val_df==-1.0).any())
x=0
for i in range(52):
    idx=random.randint(10,60)
    idx=idx+x
    train_df[idx:idx+60]=np.nan
    x=idx+60
x=0
for i in range(18):
    idx=random.randint(10,50)
    idx=idx+x
    val_df[idx:idx+60]=np.nan
    x=idx+60
x=0
for i in range(18):
    idx=random.randint(10,60)
    idx=idx+x
    test_df[idx:idx+60]=np.nan
    x=idx+60



np.save(os.path.join(opt.expPATH,'batadal_gap60_train.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'batadal_gap60_test.npy'), test_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'batadal_gap60_val.npy'), val_df, allow_pickle=False)

