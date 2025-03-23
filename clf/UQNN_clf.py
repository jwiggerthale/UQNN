#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:05:51 2025

@author: jwiggerthale
"""

#Import libraries
import os
import numpy as np
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from lenet import LeNet5
from datetime import datetime as dt


'''
Class which implements a UANN for multi class classification
Initialize with: 
    num_classes: int = 10, --> number of classes to be identified, adapt to your dataset
    lambda_u: float = 0.2, --> factor to multiply uncertainty loss (see function combibed_loss)
    lambda_c: float = 0.4, --> factor to multiply classificatuion loss (see function combibed_loss)
    num_samples: int = 10000, --> Number of forward passes conducted with MC dropout
    dropout_rate: float = 0.1, --> dropout rate of dropout layers
    file_path: str = 'Model' --> path where models and model definition are stored
'''
class UQNN(nn.Module):
    def __init__(self, 
                 num_classes: int = 10,
                 lambda_u: float = 0.2, 
                 lambda_c: float = 0.4, 
                num_samples: int = 10000,
                dropout_rate: float = 0.1,
                file_path: str = 'Model'):
        super(UQNN, self).__init__()

        #Define model structure
        #--> feature extractor = CNN with dropout, heads = fully conneted layers
        self.feature_extractor = LeNet5()
        self.classification_head = nn.Sequential(nn.Linear(84, 60), 
                                                 nn.ReLU(), 
                                                 nn.Linear(60, num_classes), 
                                                 nn.Softmax()
                                                 )
        
        self.uncertainty_head = nn.Sequential(
                                                        nn.Linear(84, 80),
                                                        nn.ReLU(),
                                                        nn.Linear(80, 60),
                                                        nn.ReLU(),
                                                        nn.Linear(60, num_classes)
                                                    )
        
        #Define optimizers for single training stages
        #--> we use Adam in both cases, for pre-training, optimizer only optizes parameters of feature extractor and classificatio head
        clf_params = list(self.feature_extractor.parameters()) + list(self.classification_head.parameters())
        self.clf_optim = optim.Adam(clf_params, lr=0.001)
        self.uncertainty_optim = optim.Adam(self.parameters(), lr=0.002)

        #Parameters for monitoring training
        self.is_converged = False
        self.clf_losses = []
        self.combined_losses = []

        #Parameters for training
        self.num_samples = num_samples
        self.lambda_u = lambda_u
        self.lambda_c = lambda_c
        
        #Write architecture of model to .txt-file
        wd = os.getcwd()
        self.file_path = f'{wd}/{file_path}'
        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)
        self.write_params()
        
        
    '''
    Function which writes parameters of model in  .txt-file
    Automatically called when initializing UQNN
    --> useful for reconstructing model when you test different configurations
    '''    
    def write_params(self):
        with open(f'{self.file_path}/definition.txt', 'a') as f:
            for attribute, value in self.__dict__.items():
                f.write(f'{attribute}: {value}\n')



    '''
    Function which determines if classifier converged
    Automatically called during training
    Works based on variation in last losses
    Minimum of 3 epochs necessary
        --> may be reduced if model converges quickly
    '''        
    def classifier_converged(self):
        if(len(self.clf_losses) < 3):
            self.is_converged = False
        elif(len(self.combined_losses) > 0):
            self.is_converged = True
        else:
            min_loss = np.min(self.clf_losses[-3:])
            max_loss = np.max(self.clf_losses[-3:])
            variation = (max_loss - min_loss)/min_loss
            if(variation > 0.05):
                self.is_converged = False
            else:
                self.is_converged = True
            

    '''
    Function which calculates how uncertain the model is regarding a certain prediction
    Within function:
        uqnn.num_samples forward passes are conducted
        mean features and mean prediction are calculated
        variance from forward passes is calculated
        uncertainty head predicts uncertainty using mean features
    Returns prediction, estimated uncertainty (from uncertainty head) but also variance from MC dropout
    Call with: 
        x: torch.tensor --> sample to be classified
    '''
    def get_uncertainty_metrics(self, 
                                x):
        self.train()
        outputs = []
        last_layers = []
        for _ in range(self.num_samples):
            output, last_layer = self.single_pass(x)
            outputs.append(output)
            last_layers.append(last_layer)
        outputs = torch.stack(outputs)
        last_layers = torch.stack(last_layers)
        last_layer = last_layers.mean(dim = 0)
        mu = outputs.mean(dim = 0)
        sigma = outputs.std(dim = 0)
        uncertainty = F.relu(self.uncertainty_head(last_layer))
        return mu, sigma, uncertainty

    '''
    Loss function for main training stage
    Automatically called in training
    Called with: 
        mu: mean prediction from MC dropout
        label: actual label
        sigma: variance from MC dropout
        uncertainty: uncertainty estiation from uncertainty head
    Returns:
        combined loss as weighted sum of cross entropy loss for classification and mse loss for uncertainty estimation
    '''
    def combined_loss(self, mu, label, sigma, uncertainty):
        clf_loss = F.cross_entropy(mu, label.long()) * self.lambda_c
        uncertainty_loss = F.mse_loss(uncertainty, sigma) * self.lambda_u
        #reg_loss = torch.mean(uncertainty) * 0.01
        return(uncertainty_loss + clf_loss).mean() 
    
    
    '''
    Function which generates data that can be used for assessment of model performance
    If required, confusin matrix for classification and performance of uncertainty estimation are plotted
    Call with: 
        data_loader: DataLoader containing the data you want to use for testing
        plot: boolean indicating if you want to see the performance plot
    '''
    def test_performance(self, 
                       data_loader: DataLoader, 
                        plot: bool = True):
        self.feature_extractor.train()  # Keep dropout active for MC Dropout
        mus = []
        sigmas = []
        uncertainties = []
        labels = []
        predictions = []
        with torch.no_grad():
            for x, y in iter(data_loader):
                pred, sigma, uncertainty = self.get_uncertainty_metrics(x)
                sigmas.extend(list(sigma.numpy()))
                classes = torch.argmax(pred, axis = 1)
                mus.extend(list(pred.cpu().detach().numpy()))
                labels.extend([elem.item() for elem in y])
                predictions.extend([elem.item() for elem in classes])
                uncertainties.extend(list(uncertainty.detach().numpy().reshape(-1)))
            if(plot):
                cm = confusion_matrix(labels, predictions)
                plot = ConfusionMatrixDisplay(cm, display_labels = np.arange(0,10))
                plot.plot()
                fig, axes = plt.subplots( figsize = (20, 20))
                axes.scatter(sigmas, uncertainties)
                min_val_1 = np.min(sigmas)
                min_val_2 = np.min(uncertainties)
                min_val = np.min([[min_val_1, min_val_2]])
                max_val_1 = np.max(sigmas)
                max_val_2 = np.max(uncertainties)
                max_val = np.max([[max_val_1, max_val_2]])
                plt.plot([min_val, max_val], [min_val, max_val], color = 'red')
                axes.set_title('Actual (x) and predicted (y) uncertainty')
            return labels, mus, sigmas, uncertainties
    
    '''
    Function to make prediction using MC dropout
    Not really necessary for the model
    May be useful if you want to test something or apply MC dropout instead of uncertaity head
    Call with: 
        x --> batch of samples to be classified 
    Returns: 
        mu: prediction for each sample
        sigma: uncertainty for each prediction
    '''
    def mc_predict(self, 
                   x):
        self.train()
        outputs = []
        last_layers = []
        for _ in range(self.num_samples):
            output, last_layer = self.single_pass(x)
            outputs.append(output)
            last_layers.append(last_layer)
        outputs = torch.stack(outputs)
        last_layers = torch.stack(last_layers)
        last_layer = last_layers.mean(dim = 0)
        mu = outputs.mean(dim = 0)
        sigma = outputs.std(dim = 0)
        return mu, sigma
    
    
    '''
    Function which trains the model
    Call with: 
        train_loader: Dataloader --> DataLoader containing the train set
        test_loader: Dataloader --> DataLoader containing the test set
        num_epochs: int --> number of epochs 
    Function 
        Conducts pre-training
        Conducts main training
        Saves model if performance is sufficient
        Writes performance in console after every epoch
    '''
    def train_model(self, 
              train_loader,
              test_loader,
              num_epochs: int = 50):
        best_loss_clf = np.inf
        best_acc_clf = 0
        best_acc_combined = 0
        best_loss_combined = np.inf
        for epoch in range(num_epochs):
            tik = dt.now()
            train_loss = 0.0
            test_loss = 0.0
            acc = 0.0
            if(self.is_converged == False):
                self.classifier_converged()
            if(self.is_converged == False):
                print(f'epoch {epoch} - classifier not yet converged')
                for x, y  in iter(train_loader):
                    pred, _ = self.single_pass(x)
                    loss = F.cross_entropy(pred, y)
                    train_loss += loss.item()
                    self.clf_optim.zero_grad()
                    loss.backward()
                    self.clf_optim.step()
                for x, y in iter(test_loader):
                    with torch.no_grad():
                        pred, _ = self.single_pass(x)
                        loss = F.cross_entropy(pred, y)
                        test_loss += loss.item()
                        classes = [torch.argmax(elem).item() for elem in pred]
                        acc+= (torch.tensor(classes) == y).float().mean()
                test_loss /= len(test_loader)
                train_loss /= len(train_loader)
                acc/= len(test_loader)
                if(test_loss < best_loss_clf):
                    torch.save(self.state_dict(), f'{self.file_path}/clf_epoch_{epoch}_loss_{int(test_loss * 100)}_acc_{int(acc*100)}.pth')
                    best_loss_clf = test_loss
                elif(acc > best_acc_clf):
                    torch.save(self.state_dict(), f'{self.file_path}/clf_epoch_{epoch}_loss_{int(test_loss * 100)}_acc_{int(acc*100)}.pth')
                    best_acc_clf = acc
                    
                print(f'Epoch {epoch}\nAccuracy: {int(acc * 100)}\nValidation Loss: {test_loss}\nTrain Loss: {train_loss}')
                self.clf_losses.append(test_loss)
            else:
                print(f'epoch {epoch} - Classifier converged')
                
                for x, y in iter(train_loader):
                    mu, sigma, uncertainty = self.get_uncertainty_metrics(x)
                    loss = self.combined_loss(mu, y, sigma, uncertainty)
                    train_loss += loss.item()
                    self.uncertainty_optim.zero_grad()
                    loss.backward()
                    self.uncertainty_optim.step()
                train_loss /= len(train_loader)
                
                uncertainties = []
                sigmas = []
                with torch.no_grad():
                    for x, y in iter(train_loader):
                        mu, sigma, uncertainty = self.get_uncertainty_metrics(x)
                        loss = self.combined_loss(mu, y, sigma, uncertainty)
                        #print(sigma)
                        #print(uncertainty)
                        sigmas.extend(list(sigma.detach().numpy().reshape(-1)))
                        uncertainties.extend(list(uncertainty.detach().numpy().reshape(-1)))
                        classes = [torch.argmax(elem).item() for elem in mu]
                        #loss = F.mse_loss(sigma, uncertainty)
                        test_loss += loss.item()
                        acc+= (torch.tensor(classes) == y).float().mean()
                test_loss /= len(test_loader)
                acc /= len(test_loader)
                ev_uncertainty = explained_variance_score(sigmas, uncertainties)
                
                if(test_loss < best_loss_combined):
                    torch.save(self.state_dict(), f'{self.file_path}/combined_epoch_{epoch}__acc_{int(acc*100)}.pth')
                    best_loss_combined = test_loss
                elif(acc > best_acc_combined):
                    torch.save(self.state_dict(), f'{self.file_path}/combined_epoch_{epoch}_loss_{int(test_loss * 100)}_acc_{int(acc*100)}.pth')
                    best_acc_combined = acc
                
                print(f'Accuracy: {int(acc * 100)}\nExplained Variance: {ev_uncertainty}\nValidation Loss: {test_loss}\nTrain Loss: {train_loss}') 
                self.combined_losses.append(test_loss)
            tok = dt.now()
            print(f'Epoch {epoch}: Took {tok - tik}\n\n')
    
    '''
    Forward pass of model
    Makes prediction on data point and estimates uncertainty
    Call with: 
        x --> batch of samples to be classified
    Returns: 
        pred --> prediction on batch 
        uncertainty --> uncertainty for predictions
    '''
    def forward(self, x):
        pred, features = self.single_pass(x)
        uncertainty = F.relu(self.uncertainty_head(features))
        return pred, uncertainty.squeeze()
        
    '''
    Function which predicts label for batch of samples and features that can be used by classification head
    Necessary in different other functions
        Call with: 
        x --> batch of samples to be classified
    Returns: 
        pred --> prediction on batch 
        features --> features extracted by the feature extractor
    '''
    def single_pass(self, x):
        features = self.feature_extractor(x)
        pred = self.classification_head(features)
        return pred, features
    
    
