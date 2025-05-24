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

parser.add_argument("--expPATH", type=str, default=os.path.expanduser('C4T-RM/SMD/smd/'),
                    help="Experiment path")
opt = parser.parse_args()
df = np.loadtxt('ServerMachineDataset/train/machine-1-1.txt',delimiter=',')
df=df[0:28470]
# Train/test split
train_df=df[0:16470]
val_df=df[16470:22470]
test_df=df[22470:]
print(test_df.shape)
print(val_df.shape)
random.seed(123)

sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
val_df = sc.transform(val_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
val_df = val_df.astype(np.float32)
test_df = test_df.astype(np.float32)

np.save(os.path.join(opt.expPATH, 'smd_gap10_val_ground_truth.npy'), val_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'smd_gap10_test_ground_truth.npy'), test_df, allow_pickle=False)

x=0
for i in range(165):
    idx=random.randint(10,60)
    idx=idx+x
    train_df[idx:idx+10]=np.nan
    x=idx+10
x=0
for i in range(60):
    idx=random.randint(10,60)
    idx=idx+x
    val_df[idx:idx+10]=np.nan
    x=idx+10
x=0
for i in range(60):
    idx=random.randint(10,60)
    idx=idx+x
    test_df[idx:idx+10]=np.nan
    x=idx+10

np.save(os.path.join(opt.expPATH,'smd_gap10_train.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'smd_gap10_test.npy'), test_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'smd_gap10_val.npy'), val_df, allow_pickle=False)

