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

parser.add_argument("--expPATH", type=str, default=os.path.expanduser('C4T-RM/SWAT/swat_point80/'),
                    help="Experiment path")

opt = parser.parse_args()

df = pd.read_csv('SWaT_Dataset_Normal_v1.csv')
df=df.drop([' Timestamp'], axis=1)
df=df.drop(['n'], axis=1)
df=df.drop(['P601'], axis=1)
df=df.drop(['P603'], axis=1)
pd.set_option('display.max_columns', None)

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
np.save(os.path.join(opt.expPATH, 'swat_point80_val_ground_truth.npy'), val_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'swat_point80_test_ground_truth.npy'), test_df, allow_pickle=False)

def point_missing(T):

    total_elements = T.shape[0] * T.shape[1]

    n=T.shape[0] * T.shape[1]*0.8

    random_indices = np.random.choice(np.arange(total_elements), int(n), replace=False)

    random_rows = random_indices // T.shape[1]
    random_cols = random_indices % T.shape[1]

    T[random_rows, random_cols] = np.nan
    return T
train_df=point_missing(train_df)
val_df=point_missing(val_df)
test_df=point_missing(test_df)


np.save(os.path.join(opt.expPATH,'swat_point80_train.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'swat_point80_test.npy'), test_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'swat_point80_val.npy'), val_df, allow_pickle=False)
