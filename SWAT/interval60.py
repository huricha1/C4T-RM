import argparse
import copy
import os
import random
from mlp import MultiLayer
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                                  GradientBoostingClassifier)
from sklearn import metrics

parser = argparse.ArgumentParser()


parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--window_size", type=int, default=30, help="size of the window")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="l2 regularization")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--cuda", type=bool, default=True,
                    help="CUDA activation")
parser.add_argument("--multiplegpu", type=bool, default=True,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs in case of multiple GPU")


parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--epoch_time_show", type=bool, default=True, help="interval betwen image samples")
parser.add_argument("--epoch_save_model_freq", type=int, default=1, help="number of epops per model save")
parser.add_argument("--minibatch_averaging", type=bool, default=True, help="Minibatch averaging")

parser.add_argument("--training", type=bool, default=True, help="Training status")
parser.add_argument("--resume", type=bool, default=False, help="Training status")
parser.add_argument("--finetuning", type=bool, default=False, help="Training status")
parser.add_argument("--generate", type=bool, default=True, help="Generating Sythetic Data")
parser.add_argument("--evaluate", type=bool, default=False, help="Evaluation status")
parser.add_argument("--PATH", type=str, default=os.path.expanduser('C4T-RM/SWAT/swat_interval60/'),
                    help="Training status")
opt = parser.parse_args()
print(opt)

seed_value = 3407
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

# Create the experiments path
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir(opt.PATH)


########## Dataset Processing ###########

class Dataset:
    def __init__(self, data, transform=None):

        # Transform
        self.transform = transform

        # load data here
        self.data = data
        self.sampleSize = len(data)
        self.featureSize = data[0].shape[1]

    def return_data(self):
        return self.data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]


        if self.transform:
            pass

        return torch.from_numpy(sample)


trainData = np.load(
    f'C4T-RM/SWAT/swat_interval60/swat_interval60_Train.npy',
    allow_pickle=False)
testData = np.load(
    f'C4T-RM/SWAT/swat_interval60/swat_interval60_Test.npy',
    allow_pickle=False)
valData = np.load(
    f'C4T-RM/SWAT/swat_interval60/swat_interval60_val.npy',
    allow_pickle=False)
test_ground_truth = np.load(
    f'C4T-RM/SWAT/swat_interval60/swat_interval60_test_ground_truth.npy',
    allow_pickle=False)
val_ground_truth = np.load(
    f'C4T-RM/SWAT/swat_interval60/swat_interval60_val_ground_truth.npy',
    allow_pickle=False)
test_miss_count = np.sum(np.isnan(testData))
val_miss_count = np.sum(np.isnan(valData))
trainData = np.nan_to_num(trainData,nan=-1)
testData = np.nan_to_num(testData,nan=-1)
valData = np.nan_to_num(valData,nan=-1)
# add channel
Data=np.concatenate((trainData, valData,testData), axis=0)
Data = np.expand_dims(Data, axis=-1)
channel_list = [Data]

# mask = torch.rand(Data.shape)
# mask[Data!= -1] = 1
# mask[Data == -1] = 0
# mask=mask.numpy()
# channel_list.append(mask)

tow = [(i % opt.window_size)/opt.window_size for i in range(Data.shape[0])]
tow = np.array(tow)
tow_tile = np.tile(tow, [1, Data.shape[1], 1]).transpose((2, 1, 0))
channel_list.append(tow_tile)

dow = [((i // opt.window_size) % opt.batch_size)/opt.batch_size for i in range(Data.shape[0])]
dow = np.array(dow)
dow_tile = np.tile(dow, [1, Data.shape[1], 1]).transpose((2, 1, 0))
channel_list.append(dow_tile)

processed_data = np.concatenate(channel_list, axis=-1)

trainData=processed_data[0:trainData.shape[0]].astype(np.float32).copy()
valData=processed_data[297000:396000].astype(np.float32).copy()
testData=processed_data[396000:].astype(np.float32).copy()
samples_list = list()
for i in range(0,trainData.shape[0]-opt.window_size+1):
    samples_list.append(trainData[i:i+opt.window_size])
# Train data loader
dataset_train_object = Dataset(data=samples_list, transform=False)

samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
dataloader_train = DataLoader(dataset_train_object, batch_size=opt.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
# Check cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device BUT it is not in use...")

# Activate CUDA
device = torch.device("cuda:0" if opt.cuda else "cpu")


class Generator_x(nn.Module):
    def __init__(self):
        super(Generator_x, self).__init__()
        # channel embeddings
        self.channel_emb = nn.Parameter(torch.empty(Data.shape[1], 30))
        nn.init.xavier_uniform_(self.channel_emb)
        # temporal embeddings
        self.time_in_window = nn.Parameter(torch.empty(opt.window_size, 30))
        nn.init.xavier_uniform_(self.time_in_window)

        self.window_in_batch = nn.Parameter(torch.empty(opt.batch_size, 30))
        nn.init.xavier_uniform_(self.window_in_batch)

        # # embedding
        self.time_series_layer = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=True)
        self.lstm0 = nn.LSTM(input_size=49, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=16, stride=1)
        # embedding
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=91, out_channels=32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=True)
        )
        # # embedding

        self.lstm = nn.LSTM(input_size=49, hidden_size=128, num_layers=1, batch_first=True)

        self.main = nn.Sequential(

            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=16, stride=2),
            nn.BatchNorm1d(opt.window_size, eps=0.001, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=9, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x, up):
        # temporal embeddings
        t_i_w_data = x[..., 1]

        time_in_window_emb = self.time_in_window[(t_i_w_data * opt.window_size).type(torch.LongTensor)]
        w_i_b_data = x[..., 2]

        window_in_batch_emb = self.window_in_batch[(w_i_b_data * opt.batch_size).type(torch.LongTensor)]
        # time series
        batch_size, len, nodes, channel = x.shape

        time_series_emb = self.time_series_layer(x.transpose(1, 3))
        time_series_emb, (h, c) = self.lstm0(time_series_emb.transpose(1, 3).squeeze(-1))
        channel_emb = []
        # channel embeddings
        channel_emb.append(
            self.channel_emb.unsqueeze(0).expand(len, -1, -1).unsqueeze(0).expand(batch_size, -1, -1, -1))
        # temporal embeddings
        tem_emb = []

        tem_emb.append(time_in_window_emb)

        tem_emb.append(window_in_batch_emb)
        # concate embeddings
        hidden = torch.cat([self.linear(time_series_emb).unsqueeze(-1)] + channel_emb + tem_emb, dim=-1)

        out = self.encode(hidden.transpose(1, 3))
        out, (h, c) = self.lstm(out.transpose(1, 3).squeeze(-1))
        out = self.main(out)
        return out

class Encoder_x(nn.Module):

    def __init__(self):
        super(Encoder_x, self).__init__()
        self.lstm = nn.LSTM(input_size=49, hidden_size=100, num_layers=2, batch_first=True)
        self.main = nn.Sequential(

            nn.Linear(100, 64),
            nn.BatchNorm1d(opt.window_size, eps=0.001, momentum=0.01),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 49),
            nn.Tanh()
        )

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.main(out)
        return out


class Discriminator_x(nn.Module):
    def __init__(self):
        super(Discriminator_x, self).__init__()
        self.conv1 = nn.Sequential(
            nn.LSTM(input_size=49, hidden_size=32, num_layers=1, batch_first=True)
        )
        self.conv2 = nn.Sequential(
            nn.LSTM(input_size=49, hidden_size=32, num_layers=1, batch_first=True)
        )

        self.conv4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.BatchNorm1d(opt.window_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        x_out, (h, c) = self.conv1(x)
        z_out, (h, c) = self.conv2(z)
        out = torch.cat([x_out, z_out], dim=2)

        out = self.conv4(out)
        return out


class Discriminator_i(nn.Module):
    def __init__(self):
        super(Discriminator_i, self).__init__()

        self.conv1 = nn.Sequential(
            nn.LSTM(input_size=49, hidden_size=100, num_layers=2, batch_first=True)
        )

        self.conv4 = nn.Sequential(

            nn.Linear(100, 64),
            nn.BatchNorm1d(opt.window_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 49),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_out, (h, c) = self.conv1(x)
        out = self.conv4(x_out)
        return out


criterion = nn.BCELoss()


def mse_loss(x_output, y_target):
    loss = nn.MSELoss(reduction='sum')
    l = loss(x_output, y_target)
    return l

def mask_data(data, mask, tau):
    return mask * data + (1 - mask) * tau

def impute_data(data, mask, tau):
    return (1-mask) * data + mask * tau

def weights_init(m):
    """
    Custom weight initialization.
    :param m: Input argument to extract layer type
    :return: Initialized architecture
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.LSTMCell:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Initialize generator and discriminator
generatorModel_x = Generator_x()
discriminatorModel_i = Discriminator_i()
discriminatorModel_x = Discriminator_x()
EncoderModel_x=Encoder_x()


if torch.cuda.device_count() > 1 and opt.multiplegpu:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  generatorModel_x = nn.DataParallel(generatorModel_x, list(range(opt.num_gpu)))
  discriminatorModel_i = nn.DataParallel(discriminatorModel_i, list(range(opt.num_gpu)))
  discriminatorModel_x = nn.DataParallel(discriminatorModel_x, list(range(opt.num_gpu)))
  EncoderModel_x = nn.DataParallel(EncoderModel_x, list(range(opt.num_gpu)))

# Put models on the allocated device
generatorModel_x.to(device)
discriminatorModel_i.to(device)
discriminatorModel_x.to(device)
EncoderModel_x.to(device)

# Weight initialization
generatorModel_x.apply(weights_init)
discriminatorModel_i.apply(weights_init)
discriminatorModel_x.apply(weights_init)
EncoderModel_x.apply(weights_init)

# Optimizers
optimizer_Gx = torch.optim.Adam(generatorModel_x.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

optimizer_Di = torch.optim.Adam(discriminatorModel_i.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
optimizer_Dx = torch.optim.Adam(discriminatorModel_x.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)
optimizer_Ex = torch.optim.Adam(EncoderModel_x.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),
                               weight_decay=opt.weight_decay)

# Define cuda Tensor
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

if opt.training:
    gen_iterations = 0
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()
        for i_batch, samples in enumerate(dataloader_train):
            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0]*samples.shape[1]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0]*samples.shape[1]).fill_(0.0), requires_grad=False)
            # Configure input
            real_samples = Variable(samples.type(Tensor)).to(device)
            real_mask = torch.rand(real_samples[...,0].shape)
            real_mask[real_samples[...,0]!= -1] = 1
            real_mask[real_samples[...,0]== -1] = 0
            real_mask = Variable(real_mask).to(device)
            # Sample noise as generator input
            z = torch.randn(real_samples[...,0].shape,device=device)
            real_samples[...,0]=mask_data(real_samples[...,0],real_mask,z)
            real_mask_data=real_samples
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # reset gradients of discriminator
            optimizer_Dx.zero_grad()

            for p in discriminatorModel_i.parameters():  # reset requires_grad
                p.requires_grad = True
            for p in discriminatorModel_x.parameters():  # reset requires_grad
                p.requires_grad = True

            fake_samples = generatorModel_x(real_mask_data,1)
            fake_mask_samples = mask_data(real_samples[...,0], real_mask, fake_samples)

            z_real = EncoderModel_x(fake_mask_samples)
            z_mask_real = mask_data(z_real, real_mask, z)
            out_realx = discriminatorModel_x(fake_mask_samples.detach(), z_mask_real.detach()).view(-1)
            realx_loss = criterion(out_realx, valid)

            out_fakex = discriminatorModel_x(fake_samples.detach(), real_mask_data[...,0]).view(-1)

            fakex_loss = criterion(out_fakex, fake)
            # total loss and calculate the backprop
            dx_loss = (realx_loss + fakex_loss)


            dx_loss.backward()
            optimizer_Dx.step()

            optimizer_Di.zero_grad()

            out_reali = discriminatorModel_i(fake_mask_samples.detach())
            reali_loss = criterion(out_reali, real_mask)
            # total loss and calculate the backprop
            reali_loss.backward()
            optimizer_Di.step()
            # -----------------
            #  Train Generator
            # -----------------

            for p in discriminatorModel_x.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in discriminatorModel_i.parameters():  # reset requires_grad
                p.requires_grad = False

            # Zero grads
            optimizer_Gx.zero_grad()

            fake_samples = generatorModel_x(real_mask_data,1)
            fake_mask_samples = mask_data(real_samples[...,0], real_mask, fake_samples)
            one_matrix = Variable(Tensor(real_samples[...,0].shape).fill_(1.0), requires_grad=False)
            dx = discriminatorModel_i(fake_mask_samples)
            gi_loss = criterion(dx, one_matrix)

            dx=discriminatorModel_x(fake_samples,real_mask_data[...,0]).view(-1)
            gx_loss = criterion(dx, valid)
            mse = mse_loss(fake_samples, fake_mask_samples)/real_mask.sum()

            loss=gi_loss+gx_loss+20*mse
            loss.backward()
            optimizer_Gx.step()

            optimizer_Ex.zero_grad()
            fake_samples = generatorModel_x(real_mask_data,1)
            fake_mask_samples = mask_data(real_samples[...,0], real_mask, fake_samples)
            z_real = EncoderModel_x(fake_mask_samples)
            z_mask_real = mask_data(z_real, real_mask,z)
            dz = discriminatorModel_x(fake_mask_samples, z_mask_real).view(-1)
            ex_loss = criterion(dz, fake)
            ex_loss.backward()
            optimizer_Ex.step()

            gen_iterations += 1
            batches_done = epoch * len(dataloader_train) + i_batch + 1
            if batches_done % opt.sample_interval == 0:
                print(
                    'TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_Dx: %.6f  Loss_Di: %.6f Loss_Gx: %.6f  Loss_Gi: %.6f  Loss_E: %.6f Loss_Rec: %.6f'
                    % (epoch + 1, opt.n_epochs, i_batch + 1, len(dataloader_train), dx_loss.item(), reali_loss.item(),gx_loss.item(), gi_loss.item(), ex_loss.item(), mse.item()), flush=True)
        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0:

            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel_x.state_dict(),
                'optimizer_G_state_dict': optimizer_Gx.state_dict(),
            }, os.path.join(opt.PATH, "model_epoch_%d.pth" % (epoch + 1)))


# if opt.generate:
for j in range(1,101):
    # Check cuda
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device BUT it is not in use...")

    # Activate CUDA
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    #####################################
    #### Load model and optimizer #######
    #####################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.PATH, f'model_epoch_{j}.pth'))

    # Load models
    generatorModel_x.load_state_dict(checkpoint['Generator_state_dict'])
    # insert weights [required]
    generatorModel_x.eval()

    # # Load val data
    num = valData.shape[0]
    gen_samples = np.zeros([num, 49])
    stride = 1
    window_size = opt.window_size

    accumulated_samples = np.zeros_like(gen_samples)
    accumulation_counts = np.zeros([num])

    start_idx = 0
    end_idx = start_idx + window_size

    while end_idx <= num:
        val = valData[start_idx:end_idx].copy()
        val = torch.from_numpy(val).to(device)
        val = val.view(1, window_size, val.shape[1],3)

        real_mask = torch.rand(val[...,0].shape)
        real_mask[val[...,0] != -1] = 1
        real_mask[val[...,0] == -1] = 0
        real_mask = Variable(real_mask).to(device)
        # Sample noise as generator input
        z = torch.randn(real_mask.shape, device=device)
        val[...,0] = mask_data(val[...,0], real_mask, z)
        impute=val
        impu_data1 = generatorModel_x(impute,0)
        impute_data = mask_data(val[...,0], real_mask, impu_data1)

        accumulated_samples[start_idx:end_idx, :] += torch.squeeze(impute_data).cpu().data.numpy()
        accumulation_counts[start_idx:end_idx] += 1

        start_idx += stride
        end_idx = start_idx + window_size

    accumulation_counts=torch.from_numpy(accumulation_counts)
    expanded_tensor = torch.repeat_interleave(accumulation_counts, repeats=accumulated_samples.shape[1], dim=0)

    accumulation_counts = expanded_tensor.view(accumulated_samples.shape[0], accumulated_samples.shape[1])
    accumulation_counts=accumulation_counts.numpy()
    gen_samples = np.divide(accumulated_samples, accumulation_counts)
    # Trasnform Object array to float
    gen_samples = gen_samples.astype(np.float32)
    np.save(os.path.join(opt.PATH, f'syntheticlabel{j}.npy'), gen_samples, allow_pickle=False)


 #### test #######
# Loading the checkpoint
checkpoint = torch.load(os.path.join(opt.PATH, f'model_epoch_.pth'))

# Load models
generatorModel_x.load_state_dict(checkpoint['Generator_state_dict'])

generatorModel_x.eval()

# # Load test data
num = testData.shape[0]
gen_samples = np.zeros([num, 49])
stride = 1
window_size = opt.window_size
accumulated_samples = np.zeros_like(gen_samples)

accumulation_counts = np.zeros([num])

start_idx = 0
end_idx = start_idx + window_size

while end_idx <= num:
    test = testData[start_idx:end_idx].copy()
    test = torch.from_numpy(test).to(device)
    test = test.view(1, window_size, test.shape[1],3)

    real_mask = torch.rand(test[...,0].shape)
    real_mask[test[...,0] != -1] = 1
    real_mask[test[...,0] == -1] = 0
    real_mask = Variable(real_mask).to(device)
    # Sample noise as generator input
    z = torch.randn(real_mask.shape, device=device)
    test[...,0] = mask_data(test[...,0], real_mask, z)
    impute=test
    impu_data1 = generatorModel_x(impute,0)
    impute_data = mask_data(test[...,0], real_mask, impu_data1)
    accumulated_samples[start_idx:end_idx, :] += torch.squeeze(impute_data).cpu().data.numpy()
    accumulation_counts[start_idx:end_idx] += 1

    start_idx += stride
    end_idx = start_idx + window_size

accumulation_counts = torch.from_numpy(accumulation_counts)
expanded_tensor = torch.repeat_interleave(accumulation_counts, repeats=accumulated_samples.shape[1], dim=0)

accumulation_counts = expanded_tensor.view(accumulated_samples.shape[0], accumulated_samples.shape[1])
accumulation_counts = accumulation_counts.numpy()
gen_samples = np.divide(accumulated_samples, accumulation_counts)

print('gen_samples:', gen_samples.shape[0])
# Trasnform Object array to float
gen_samples = gen_samples.astype(np.float32)
# ave synthetic data
np.save(os.path.join(opt.PATH, f'syntheticlabel.npy'), gen_samples, allow_pickle=False)


