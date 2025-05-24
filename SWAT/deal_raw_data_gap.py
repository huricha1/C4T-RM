import argparse
import numpy as np
import time
import random
import os
import sklearn.model_selection as skl
import matplotlib.pyplot as plt
import copy
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


parser.add_argument("--expPATH", type=str, default=os.path.expanduser('C4T-RM/SWAT/swat/'),
                    help="Experiment path")

opt = parser.parse_args()
df = pd.read_csv('SWaT_Dataset_Normal_v1.csv')
df=df.drop([' Timestamp'], axis=1)
df=df.drop(['n'], axis=1)
df=df.drop(['P601'], axis=1)
df=df.drop(['P603'], axis=1)
pd.set_option('display.max_columns', None)
print(df.shape)
# Train/test split
train_df=df[0:297000]
val_df=df[297000:396000]
test_df=df[396000:]
random.seed(123)

sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
val_df = sc.transform(val_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
val_df = val_df.astype(np.float32)
test_df = test_df.astype(np.float32)

test_df_copy=copy.deepcopy(test_df)
np.save(os.path.join(opt.expPATH, 'swat_gap30_val_ground_truth.npy'), val_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'swat_gap30_test_ground_truth.npy'), test_df, allow_pickle=False)

x=0
for i in range(2970):
    idx=random.randint(10,100)
    idx=idx+x
    train_df[idx:idx+30]=np.nan
    x=idx+30
x=0
for i in range(990):
    idx = random.randint(10, 100)
    idx = idx + x
    val_df[idx:idx + 30] = np.nan
    x = idx + 30
x=0
for i in range(990):
    idx = random.randint(10, 100)
    idx = idx + x
    test_df[idx:idx + 30] = np.nan
    x = idx + 30




np.save(os.path.join(opt.expPATH,'swat_gap30_train.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'swat_gap30_test.npy'), test_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'swat_gap30_val.npy'), val_df, allow_pickle=False)
