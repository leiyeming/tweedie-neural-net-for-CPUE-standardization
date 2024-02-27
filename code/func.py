
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from torch.optim import Adam, SGD
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.modules.loss import MSELoss
import time
import matplotlib.pyplot as plt
import torch.distributions as dist
from tweedie import tweedie
import scipy as sp
import os
from torchmetrics.functional import tweedie_deviance_score
import matplotlib.pyplot as plt


# define the function to normalize continue variables, here lon and lat
def lon_lat_norm(lon_lat,data):
    train_stats = lon_lat.describe()
    train_stats.head()
    train_stats = train_stats.transpose()
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    return norm(data)

n2t = lambda n: torch.from_numpy(n).type(torch.float32)


# split data based on # of rows
def split_data_NPF(data, train_row=None, test_row=None, pred_row=None,pred_all_row=None):
    X_tr = data.iloc[:train_row, :]
    X_ts = data.iloc[train_row:train_row + test_row, :]
    X_pred = data.iloc[train_row + test_row:train_row + test_row + pred_row, :]

    X_tr = n2t(X_tr.values)
    X_ts = n2t(X_ts.values)
    X_pred = n2t(X_pred.values)

    if pred_all_row is not None:
        X_pred_all = data.iloc[train_row + test_row + pred_row:train_row + test_row + pred_row + pred_all_row, :]
        X_pred_all = n2t(X_pred_all.values)
        return X_tr, X_ts, X_pred, X_pred_all
    else:
        return X_tr, X_ts, X_pred

# linearly combine 3 modules
class tw_EqB(nn.Module):
    def __init__(self, *mlps):
        super(tw_EqB, self).__init__()
        self.mlps = nn.ModuleList(mlps)
        # self.linear_layer = nn.Linear(sum(mlp[-2].out_features for mlp in mlps), 1, bias=False)
        self.linear_layer = nn.Linear(len(mlps), 1)
        # self.relu = nn.Sigmoid()

    def forward(self, *inputs):
        outputs = []
        for mlp, x in zip(self.mlps, inputs):
            out = mlp(x)
            outputs.append(out)
        combined_output = torch.cat(outputs, dim=1)
        linear_output = self.linear_layer(combined_output)
        output = torch.exp(linear_output)
        return output, outputs

# define datawrapper
class DatasetWrapper_tw(Dataset):
    def __init__(self, inputs_list, labels):
        self.inputs_list = inputs_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Your data retrieval logic here
        samples = [inputs_list[idx] for inputs_list in self.inputs_list]
        lables = self.labels[idx]
        # return tuple(samples), lables
        return tuple(samples), lables




# early stopping for all architectures
class EarlyStopping:
    def __init__(self, patience=100, delta=0, verbose=False, loc=None,NN_str=None):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.verbose = verbose
        self.loc = loc
        self.NN_str=str(NN_str)

    def __call__(self, val_loss, model,p,phi):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model,p,phi)
        elif val_loss >= self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model,p,phi)
            self.counter = 0

    def get_counter(self):
        return self.counter

    def save_checkpoint(self, val_loss, model,p,phi):
        if val_loss < self.val_loss_min:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

            torch.save(model.state_dict(), self.loc + "model/" + self.NN_str + '.pt')

            data = {'p': [p], 'phi': [phi]}
            df = pd.DataFrame(data)

            # Save the DataFrame to a CSV file
            df.to_csv(self.loc + "model/" + self.NN_str + "para.csv", index=False)
            self.val_loss_min = val_loss


# training tweedie neural net         
def train_tw(net, optimizer, scheduler, loss, dataset, nepochs=100, batch_size=100, val_data=None,
                    test_data=None, test_y=None,
                   val_y=None, device=None, scenario=None, NN_str=None, early_stopping=None, criterion=None,l2_train=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    l2_lambda=0.001
    val_y = val_y.to(device)
    test_y = test_y.to(device)
    val_tensors = [item.clone().detach().to(device) for item in val_data]
    test_tensors = [item.clone().detach().to(device) for item in test_data]


    start_time = time.time()
    p=torch.tensor([1.5])
    p=p.to(device)
    phi=0
    for epoch in range(nepochs):
        ep_st=True
        if (epoch + 1) % 2 == 0:
            end_time = time.time()
            epoch_time = end_time - start_time
            # train_loss,p,phi,ep_st = tweedie_loss(mu=mu_val.squeeze(), p=p, phi=phi,target=val_y.float().squeeze(),device=device,epoch=epoch,ep_st=ep_st)
            r_squared_val = r_squared(val_y.float().squeeze(), mu_val.squeeze())
            r_squared_test = r_squared(test_y.float().squeeze(), mu_test.squeeze())
            print(
                'GPU {}, {}, Epoch [{}/{}], early_stop: {:.0f},R2_val: {:.4f},MSE_val: {:.4f},R2_test: {:.4f},MSE_test: {:.4f}, lr: {:.5f},  p: {:.5f},phi: {:.5f}, time: {:.4f}'.format(
                    device,
                    NN_str,
                    epoch + 1,
                    nepochs,
                    # loss,
                    # train_loss,
                    early_stopping.get_counter(),
                    r_squared_val,
                    val_loss,
                    r_squared_test,
                    test_loss,
                    optimizer.param_groups[0]['lr'],
                    # mu_val_2.item(),
                    p.item(),
                    phi,
                    epoch_time))
            start_time = time.time()
        for inputs, labels in dataloader:

            optimizer.zero_grad()

            # forward pass
            mu,others= net(*inputs)
            loss,p,phi,ep_st = tweedie_loss(mu=mu.squeeze(), p=p, phi=phi,target=labels.float().squeeze(),device=device,epoch=epoch,ep_st=ep_st)

            if  l2_train:
                l2_norm = sum(p.pow(2.0).sum()
                              for p in net.parameters())

                loss = loss + l2_lambda * l2_norm

            # backward pass and optimization
            loss.backward()
            optimizer.step()
        scheduler.step()
        with torch.no_grad():
            mu_val,others = net(*val_tensors)
            # val_loss = tweedie_loss(mu=mu.squeeze(), p=p, target=labels.float().squeeze())
            val_loss = criterion(mu_val.squeeze(), val_y.float().squeeze())
            # mu_val_2 = torch.mean(mu_val)
        mu_test, others = net(*test_tensors)
        test_loss = criterion(mu_test.squeeze(), test_y.float().squeeze())

        early_stopping(val_loss, net,p.item(),phi)
        if early_stopping.early_stop:
            print('Early stopping')
            break