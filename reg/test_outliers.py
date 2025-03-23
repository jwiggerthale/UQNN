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
    
import plotly.io as pio
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
    --> mean and std required for tests
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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    
torch.manual_seed(1)
np.random.seed(1)

print ("Loading data and other hyperparameters...")
# Data and related information
file_path = '/t1/erlangen/users/jwiggerthale/TestDatasets/bostonHousing/data'
data = np.loadtxt(f'{file_path}/data.txt')
index_features = np.loadtxt(f'{file_path}/index_features.txt')
n_splits = np.loadtxt(f'{file_path}/n_splits.txt')
index_target = np.loadtxt(f'{file_path}/index_target.txt')



#Get data
X = data[ : , [int(i) for i in index_features.tolist()] ]
y = data[ : , int(index_target.tolist()) ]

split = 0
index_train = np.loadtxt(_get_index_train_test_path(file_path, split, train=True))
index_test = np.loadtxt(_get_index_train_test_path(file_path, split, train=False))

x_train = X[ [int(i) for i in index_train.tolist()] ]
y_train = y[ [int(i) for i in index_train.tolist()] ]
X_test = X[ [int(i) for i in index_test.tolist()] ]
y_test = y[ [int(i) for i in index_test.tolist()] ]



#Get mean and std for x and y
_, _, y_train_mean, y_train_std, x_train_mean, x_train_std = get_dataloader(x_train, y_train)



#Get outliers and put them in DataLoader
data = pd.DataFrame(data)
numeric_cols = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12]

z_scores = data[numeric_cols].apply(zscore)
outlier_mask = (z_scores.abs() > 3).any(axis=1)
outliers = data[outlier_mask]
outliers = np.array(outliers)

normal = data[~outlier_mask]
normal = np.array(normal)

x_outliers = outliers[: , [int(i) for i in index_features.tolist()] ]
y_outliers = outliers[:, int(index_target.tolist())]
x_outliers = (x_outliers - x_train_mean) / x_train_std
y_outliers = (y_outliers - y_train_mean) / y_train_std

x_normal = normal[: , [int(i) for i in index_features.tolist()] ]
y_normal = normal[:, int(index_target.tolist())]
x_normal = (x_normal - x_train_mean) / x_train_std
y_normal = (y_normal - y_train_mean) / y_train_std

val_set = my_data_loader(x_outliers, y_outliers)
val_loader = DataLoader(val_set)#, batch_size = 16)
train_set = my_data_loader(x_normal, y_normal)
train_loader = DataLoader(train_set)#, batch_size = 16)

#Model information
num_epochs = 40
dropout = 0.01
tau = 0.15
n_hidden = [50, 50, 50]
lambda_u = 1.2
lambda_r = 0.4
num_samples  = 500


#Create model and load weights
my_regressor = MyRegressor(input_dim = len(X[0]), 
                 n_hidden = n_hidden, 
                 dropout_rate = dropout, 
                 tau = tau,
                 lambda_u = lambda_u, 
                 lambda_r = lambda_r, 
                 num_samples = num_samples, 
                 file_path = '/models')
  
f = './model/UQNN_clf.pth'
my_regressor.load_state_dict(torch.load(f))

#Get predictions and uncertainties on outliers
preds = []
uncertainties = []
labels = []
for x, y in iter(val_loader):
  pred, un = my_regressor.forward(x)
  preds.extend([p.item() for p in pred])
  uncertainties.extend([u.item() for u in un])
  labels.extend([l.item() for l in y])

#Create data frame for predictions and uncertainties on outliers
results = pd.DataFrame(columns = ['uncertainty', 'pred', 'label'])
results['uncertainty'] = uncertainties
results['pred'] = preds
results['label'] = labels
results['DS'] = 'outliers'


#Get predictions and uncertainties for common data
preds = []
uncertainties = []
labels = []
for x, y in iter(train_loader):
  pred, un = my_regressor.forward(x)
  preds.extend([p.item() for p in pred])
  uncertainties.extend([u.item() for u in un])
  labels.extend([l.item() for l in y])


#Create data frame for results on common data
results2 = pd.DataFrame(columns = ['uncertainty', 'pred', 'label'])
results2['uncertainty'] = uncertainties
results2['pred'] = preds
results2['label'] = labels
results2['DS'] = 'test'

#combine results on common data and outliers
results = pd.concat([results, results2])

#plot uncertainties on common data and outliers
fig = px.scatter(results,  y = 'uncertainty', color = 'DS')
fig.update_layout(title = f'Results on Common Data and Outliers - {f}')
fig.show()

#boxplot of uncertainties for common data and outliers
fig = px.box(results, x = 'DS', y = 'uncertainty')
fig.show()
        
        
