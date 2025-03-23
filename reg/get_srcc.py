#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:51:26 2025

@author: jwiggerthale
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from UncertaintyAwareRegressor import MyRegressor
from scipy.stats import zscore
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from scipy.stats import spearmanr
import plotly.io as pio

#make sure, plots render in browser
pio.renderers.default = 'browser'  


"""
Function which generate file path containing the training/test split for split number (split number from 1 - 20)
Call with: 
  file_path --> path to directry containing data
  split_num --> split to be used
  trai --> boolean indicating if data is training data
Returns: 
  path to txt file containing data
"""
def _get_index_train_test_path(file_path, split_num, train = True):
    if train:
        return file_path + "/index_train_" + str(split_num) + ".txt"
    else:
        return file_path + "/index_test_" + str(split_num) + ".txt" 
 

'''
Class for loading data 
initialize with: 
  x --> numpy array of features
  y --> numpy array of labels
  __getitem__ returns: 
    x --> features for certain sample
    y --> target belonging to features
'''
class my_data_loader(Dataset): 
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)
    def __len__(self):
        return(len(self.x))
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return(x,y)


'''
Function which created data loader for training
Call with: 
  x --> numpy array of features
  y --> numpy array of labels
  normalize --> boolean indicating if x and y shall be normalized
  batch_size --> bathc size of data loader
Returns: 
  train_loader --> data loader for training
  val_loader --> data loader for validation
  x_train_mean --> mean of x_train
  x_train_std --> std of x_train
  y_train_mean --> mean of y_train
  y_train_std --> std of y_train
    --> mean and std may be required for further tests in some cases
'''
def get_dataloader(x, y, 
                   normalize: bool = True, 
                   batch_size: int = 128):
    X_train_original = x
    y_train_original = y
    num_training_examples = int(0.8 * x.shape[0])   
    x_val = x[num_training_examples:, :]
    y_val = y[num_training_examples:]

    x_train = x[0:num_training_examples, :]
    y_train = y[0:num_training_examples]

    x_train_mean = np.mean(x_train, axis=0)    
    x_train_std = np.std(x_train, axis=0)
    x_train_std[x_train_std == 0] = 1
    
    if(normalize == True):
        x_train = (x_train - x_train_mean) / x_train_std
        x_val = (x_val - x_train_mean) / x_train_std
        
    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)
    y_train_norm = (y_train - y_train_mean)/ y_train_std
    y_val_norm = (y_val - y_train_mean)/ y_train_std
    
    train_set = my_data_loader(x_train, y_train_norm)
    val_set = my_data_loader(x_val, y_val_norm)
    val_loader = DataLoader(val_set)
    train_loader =  DataLoader(train_set)
    return train_loader, val_loader,  y_train_mean, y_train_std, x_train_mean, x_train_std



#Get data
print ("Loading data and other hyperparameters...")
# Data and related information
file_path = '/t1/erlangen/users/jwiggerthale/TestDatasets/bostonHousing/data'
data = np.loadtxt(f'{file_path}/data.txt')
index_features = np.loadtxt(f'{file_path}/index_features.txt')
n_splits = np.loadtxt(f'{file_path}/n_splits.txt')
index_target = np.loadtxt(f'{file_path}/index_target.txt')
X = data[ : , [int(i) for i in index_features.tolist()] ]
y = data[ : , int(index_target.tolist()) ]
split = 0
index_train = np.loadtxt(_get_index_train_test_path(file_path, split, train=True))
index_test = np.loadtxt(_get_index_train_test_path(file_path, split, train=False))
x_train = X[ [int(i) for i in index_train.tolist()] ]
y_train = y[ [int(i) for i in index_train.tolist()] ]
X_test = X[ [int(i) for i in index_test.tolist()] ]
y_test = y[ [int(i) for i in index_test.tolist()] ]

#Get train loader (and other parameters which are not required)
train_loader, val_loader, y_train_mean, y_train_std, x_train_mean, x_train_std = get_dataloader(x_train, y_train)

#Model information
num_epochs = 40
dropout = 0.01
tau = 0.15
n_hidden = [50, 50, 50]
lambda_u = 1.2
lambda_c = 0.4
num_samples  = 500

#Create model and load weights
my_regressor = MyRegressor(input_dim = len(X[0]), 
                 n_hidden = n_hidden, 
                 dropout_rate = dropout, 
                 tau = tau,
                 lambda_u = lambda_u, 
                 lambda_c = lambda_c, 
                 num_samples = num_samples, 
                 file_path = '/Regressor_V4')
    


f = './models/UQNN_clf.pth'
my_regressor.load_state_dict(torch.load(f))

#Reproducibility
torch.manual_seed(17)
np.random.seed(17)

#Get predictions and uncertainties from UQNN
preds_val = []
uncertainties_val = []
labels_val = []
start = datetime.now()
for x, y in iter(val_loader):
  pred, un = my_regressor.forward(x)
  preds_val.extend([p.item() for p in pred])
  uncertainties_val.extend([u.item() for u in un])
  labels_val.extend([l.item() for l in y])
end = datetime.now()
delta_uqnn = end - start
  
labels_val = np.array(labels_val)
preds_val = np.array(preds_val)
uncertainties_val = np.array(uncertainties_val)
errors_val = abs(labels_val - preds_val)

#Get predictions and uncertainties from MC dropout
start = datetime.now()
_, _, var, preds_mc, _, all_preds = my_regressor.mc_predict(val_loader)
end = datetime.now()
uncertainties_mc = np.array(all_preds).var(axis = 1)
errors_mc = abs(labels_val - preds_mc)
delta_mc = end - start

#Calculate SRCC
scrr_val = spearmanr(uncertainties_val, errors_val)
scrr_mc = spearmanr(uncertainties_mc, errors_mc)

  
#print results
print(f'seed: 17\nscrr outliers: {scrr_val} (took {delta_uqnn})\
        \n scrr inliers mc: {scrr_mc} (took {delta_mc})\n\n')

       
