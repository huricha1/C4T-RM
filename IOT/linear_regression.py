from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
import numpy as np
import torch
import h5py
dataTest_ground_truth = np.load(f'iot_gap10_test_ground_truth.npy', allow_pickle=False)
syn0=np.load(f'iot_gap10_test_ground_truth.npy', allow_pickle=False)

syn1 = np.load(f'syntheticlabel.npy', allow_pickle=False)

syn2 = np.load(f'iot_gap10_Test.npy', allow_pickle=False)
syn2 = np.nan_to_num(syn2,nan=-1)
def linear(syn,index):
    td = syn[0:720].copy()
    mask = np.any(td == -1, axis=1)
    new_arr = np.delete(td, np.where(mask), axis=0)
    X_train_r = new_arr.copy()
    X_test_r=dataTest_ground_truth[720:].copy()


    a = X_train_r[:, :index].copy()
    b = X_train_r[:, index+1:].copy()
    x_train_lr = np.concatenate((a, b), axis=1)
    y_train_lr = X_train_r[:, index].copy()

    a = X_test_r[:, :index].copy()
    b = X_test_r[:, index+1:].copy()
    x_test_lr = np.concatenate((a, b), axis=1)
    y_test_lr = X_test_r[:, index].copy()
    linear_regressor = LinearRegression()
    linear_regressor.fit(x_train_lr, y_train_lr)
    y_pre = linear_regressor.predict(x_test_lr)
    loss = nn.MSELoss(reduction='sum')
    l =np.abs(y_pre-y_test_lr)
    l=np.sum(l)
    print(len(y_test_lr))
    return l
if __name__ == '__main__':
    l0 = linear(syn0, 3)
    l1 = linear(syn1, 3)
    l2 = linear(syn2, 3)



