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

parser.add_argument("--expPATH", type=str, default=os.path.expanduser('C4T-RM/EPIC/epic/'),
                    help="Experiment path")

opt = parser.parse_args()

df = pd.read_csv('EpicLog_Scenario 6_19_Oct_2018_16_06.csv')
df=df.drop(['Timestamp'], axis=1)
df=df[0:840].copy()
# Train/test split
train_df=df[0:480].copy()
val_df=df[480:660].copy()
test_df=df[660:].copy()
print(test_df.shape,val_df.shape)
random.seed(123)

sc = MinMaxScaler()
train_df = sc.fit_transform(train_df)
val_df = sc.transform(val_df)
test_df = sc.transform(test_df)
train_df = train_df.astype(np.float32)
val_df = val_df.astype(np.float32)
test_df = test_df.astype(np.float32)

np.save(os.path.join(opt.expPATH, 'epic_interval60_val_ground_truth.npy'), val_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'epic_interval60_test_ground_truth.npy'), test_df, allow_pickle=False)

# missing rate 10%
# for i in range(1,6):
#     train_df[i:train_df.shape[0]:6,1:83:9]=np.nan
#     val_df[i:val_df.shape[0]:6, 1:83:9] = np.nan
#     test_df[i:test_df.shape[0]:6,1:83:9]=np.nan
# for i in range(1,11):
#     train_df[i:train_df.shape[0]:11,91:173:9]=np.nan
#     val_df[i:val_df.shape[0]:11, 91:173:9] = np.nan
#     test_df[i:test_df.shape[0]:11,91:173:9]=np.nan
# for i in range(1,21):
#     train_df[i:train_df.shape[0]:21,181:281:9]=np.nan
#     val_df[i:val_df.shape[0]:21, 181:281:9] = np.nan
#     test_df[i:test_df.shape[0]:21,181:281:9]=np.nan

#missing rate 30%
# for i in range(1,6):
#     train_df[i:train_df.shape[0]:6,1:89:3]=np.nan
#     val_df[i:val_df.shape[0]:6, 1:89:3] = np.nan
#     test_df[i:test_df.shape[0]:6,1:89:3]=np.nan
# for i in range(1,11):
#     train_df[i:train_df.shape[0]:11,91:179:3]=np.nan
#     val_df[i:val_df.shape[0]:11, 91:179:3] = np.nan
#     test_df[i:test_df.shape[0]:11,91:179:3]=np.nan
# for i in range(1,21):
#     train_df[i:train_df.shape[0]:21,181:290:3]=np.nan
#     val_df[i:val_df.shape[0]:21, 181:290:3] = np.nan
#     test_df[i:test_df.shape[0]:21,181:290:3]=np.nan

#missing rate 60%
for i in range(1,6):
    train_df[i:train_df.shape[0]:6,1:89:3]=np.nan
    train_df[i:train_df.shape[0]:6, 2:90:3] = np.nan
    val_df[i:val_df.shape[0]:6, 1:89:3] = np.nan
    val_df[i:val_df.shape[0]:6, 2:90:3] = np.nan
    test_df[i:test_df.shape[0]:6,1:89:3]=np.nan
    test_df[i:test_df.shape[0]:6, 2:90:3] = np.nan
for i in range(1,11):
    train_df[i:train_df.shape[0]:11,91:179:3]=np.nan
    train_df[i:train_df.shape[0]:11, 92:180:3] = np.nan
    val_df[i:val_df.shape[0]:11, 91:179:3] = np.nan
    val_df[i:val_df.shape[0]:11, 92:180:3] = np.nan
    test_df[i:test_df.shape[0]:11,91:179:3]=np.nan
    test_df[i:test_df.shape[0]:11, 92:180:3] = np.nan
for i in range(1,21):
    train_df[i:train_df.shape[0]:21,181:290:3]=np.nan
    train_df[i:train_df.shape[0]:21, 182:291:3] = np.nan
    val_df[i:val_df.shape[0]:21, 181:290:3] = np.nan
    val_df[i:val_df.shape[0]:21, 182:291:3] = np.nan
    test_df[i:test_df.shape[0]:21,181:290:3]=np.nan
    test_df[i:test_df.shape[0]:21, 182:291:3] = np.nan


np.save(os.path.join(opt.expPATH,'epic_interval60_train.npy'), train_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'epic_interval60_test.npy'), test_df, allow_pickle=False)
np.save(os.path.join(opt.expPATH, 'epic_interval60_val.npy'), val_df, allow_pickle=False)

