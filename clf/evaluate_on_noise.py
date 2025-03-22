#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 07:51:13 2025

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
from UQNN_clf import UQNN
import pandas as pd
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
 

'''
Class for mnist dataset 
For details see train_UQNN.py
'''
class mnist_dataset(Dataset): 
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize(mean=[ 0.1307],std=[0.3081]),
                             ])
    def __len__(self):
        return(len(self.data))
    def __getitem__(self, idx):
        im = self.data.iloc[idx, :-1]
        im = np.array(im).reshape((28, 28))
        im= im/255
        im = self.transform(im).double()
        label = self.data.iloc[idx, -1]
        return(im.float(), label)
    
    

#Reproducibility
torch.manual_seed(1)
np.random.seed(1)


#Get test data and noise and put into data loader
test = pd.read_csv('./data/mnist_test.csv').drop('Unnamed: 0', axis = 1)
test = test.iloc[:100]
noise = pd.read_csv('./data/noise.csv').drop('Unnamed: 0', axis = 1)

test_set = mnist_dataset(test)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)

noise_set = mnist_dataset(noise)
noise_loader = DataLoader(noise_set, batch_size = 1, shuffle = False)


#Model information
lambda_u = 0.1
lambda_c = 1.2
num_samples  = 100
num_classes = 10

my_classifier = UQNN(num_classes = num_classes,
                 lambda_u = lambda_u, 
                 lambda_c = lambda_c, 
                 num_samples = num_samples,
                 file_path = '/Clf_V4')


my_classifier.load_state_dict(torch.load('./models/UQNN_clf.pth'))
my_classifier.eval()

test_uns = []
for im, label in iter(test_loader):
    pred, un = my_classifier.forward(im)
    predicted_class = pred.argmax().item()
    test_uns.append(un[predicted_class].item())
    
noise_uns = []
for im, label in iter(noise_loader):
    pred, un = my_classifier.forward(im)
    predicted_class = pred.argmax().item()
    noise_uns.append(un[predicted_class].item())
    
    
fig, ax = plt.subplots(figsize = (12,12))
ax.scatter(np.arange(1, 101), test_uns, label = 'Test set', color = 'green')
ax.scatter(np.arange(1, 101), noise_uns, label = 'Noise', color = 'red')
ax.set_title('Uncertainty of UANN trained for MNIST Data on Test Images (Green) and Noise (Red)')
ax.set_xlabel('Index')
ax.set_ylabel('Uncertainty')
