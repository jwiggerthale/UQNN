#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 07:51:13 2025

@author: jwiggerthale
"""
'''
This script implements the rotation test described in our paper
When running the script, results will be plotted for 10 sample images
'''


#Import libraries
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
Dataset which returns image rotated in angles from 0 - 95 degree in 5 degree steps
Call with: 
  df --> DataFrame containing images and labels
__getitem__ returns: 
  list of images rotated between 0 ad 95 degree
  label
'''
class rotation_mnist_dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = df
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ])
        self.rotation_angles = range(0, 100, 5)  # 0 bis 95 Grad in 5-Grad-Schritten

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im = self.data.iloc[idx, :-1].values.astype(np.uint8)
        im = im.reshape((28, 28))
        im = Image.fromarray(im)

        rotated_images = []
        for angle in self.rotation_angles:
            rotated_im = im.rotate(angle)
            rotated_im = self.transform(rotated_im)
            rotated_images.append(rotated_im)

        label = self.data.iloc[idx, -1]
        return rotated_images, label

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

train_loader = DataLoader(train_set, batch_size = 1, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)

#Model information
lambda_u = 0.1
lambda_c = 1.2
num_samples  = 100
num_classes = 10

#Create model and load weights
my_classifier = UQNN(num_classes = num_classes,
                 lambda_u = lambda_u, 
                 lambda_c = lambda_c, 
                 num_samples = num_samples,
                 file_path = '/Clf_V4')

my_classifier.load_state_dict(torch.load('./models/UQNN_clf.pth'))
my_classifier.eval()

#Get 10 next imaes from data loader (You can adapt the number of images)
all_ims = []
for i in range(10):
    ims, label = next(iter(train_loader))
    all_ims.append(ims)
    

#Get predictions, uncertainties and list of all classes predicted at least once for an image for all images
all_preds = []
all_uncertainties = []
all_predicted_classes = []
for ims in all_ims:
    preds = []
    labels = []
    uncertainties = []
    predicted_classes = []
    for im in ims:
        pred, un = my_classifier.forward(im)
        preds.append(pred.cpu().detach().numpy())
        uncertainties.append(un)
        predicted_class = pred.argmax(dim = 1)
        predicted_classes.extend([p.item() for p in predicted_class])
    all_preds.append(preds)
    all_uncertainties.append(uncertainties)
    all_predicted_classes.append(predicted_classes)
        
    

#Create color scheme for classes
for i, predicted_classes in enumerate(all_predicted_classes):
    angles = range(0, 100, 5) 
    num_classes = 10  # Anzahl der Klassen
    classes = np.unique(predicted_classes)
    colors = plt.cm.jet(np.linspace(0, 1, num_classes))
    
    # Figure Setup
    # (b) Softmax Output Scatter (Wahrscheinlichkeiten)
    uncertainties = [u.cpu().detach().numpy() for u in all_uncertainties[i]] 
    softmax_outputs = np.array(uncertainties)  # Shape (Num_Angles, Num_Classes)
    
    #Create plot
    #left pane shows softmax outputs for classes predicted at least once
    #right pane shows uncertainty values for the corresponding classes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # (a) Softmax Input Scatter (Logits)
    for class_idx in classes:
        logits_per_class = [pred[0][class_idx].item() for pred in all_preds[i]]
        axes[0].scatter(angles, logits_per_class, color=colors[class_idx], alpha=0.5, label=f'Class {class_idx}')
        probs_per_class = softmax_outputs[:, class_idx]
        normalized_probs = probs_per_class / logits_per_class
        axes[1].scatter(angles, abs(probs_per_class), color=colors[class_idx], alpha=0.5, label=f'Class {class_idx}')
    
    
    axes[0].set_title('Predicted value for class')
    axes[0].set_xlabel("Angle")
    axes[0].set_ylabel("Predicted value")
    axes[0].legend(loc="upper right")
    
    
    axes[1].set_title("Model uncertainty")
    axes[1].set_xlabel("Angle")
    axes[1].set_ylabel("Uncertainty")
    axes[1].legend(loc="upper right")
    
    ims_to_plot = [all_ims[i][0], all_ims[i][4], all_ims[i][9], all_ims[i][14], all_ims[i][19]]
    for i, img in enumerate(ims_to_plot):
        ax = fig.add_axes([0.09 + i * 0.08, -0.1, 0.07, 0.07], anchor='NE', zorder=1)
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis("off")
        ax = fig.add_axes([0.51 + i * 0.08, -0.1, 0.07, 0.07], anchor='NE', zorder=1)
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis("off")  
    plt.show()
