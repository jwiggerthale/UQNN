#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:51:26 2025

@author: jwiggerthale
"""
'''
This script implements training of UQNN for classification
When executing script, performance will be printed in console after each epoch and models will be saved when performance is sufficient
'''

#Import libraries
import numpy as np
from torch.utils.data import DataLoader, Dataset
from UQNN_clf import UQNN
import pandas as pd
from torchvision import transforms
 

'''
Class for dataset 
Based on a pandas DataFrame which contains image in columns 0 - 28 x 28 -1 and label in column 28 x 28
  --> seems crazy but necessary since we worked in a protected environment where downloading the dataset was not possible
  --> you can use default datasets provided by PyTorch
Call with: 
  df --> DataFrame containing images and labels
Function __getitem__ returns image as torch.tensor and label as torch.tensor
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
        return(im, label)

#Reproducibility
torch.manual_seed(1)
np.random.seed(1)

#Get train set and create Dataset and DataLoader
train = pd.read_csv('./data/train_1.csv').drop('Unnamed: 0', axis = 1)
for i in range(4):
 subset = pd.read_csv(f'./data/train_{i+2}.csv').drop('Unnamed: 0', axis = 1)
 train = pd.concat([train, subset])
test = pd.read_csv('./data/mnist_test.csv').drop('Unnamed: 0', axis = 1)

train_set = mnist_dataset(train)
test_set = mnist_dataset(test)

train_loader = DataLoader(train_set, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 16, shuffle = False)


#Define information for model and training
num_epochs = 50
lambda_u = 1.2
lambda_c = 0.4
num_samples  = 100
num_classes = 10


#Create classifier and train
my_classifier = UQNN(num_classes = num_classes,
                 lambda_u = lambda_u, 
                 lambda_c = lambda_c, 
                 num_samples = num_samples, 
                 file_path = './Clf_softmax_not_pretrained')

my_classifier.double()
    

my_classifier.train_model(train_loader, test_loader, num_epochs = num_epochs)
