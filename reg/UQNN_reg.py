#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:05:51 2025

@author: jwiggerthale
"""
#Import libraries
import pandas as pd
import os
import numpy as np
from sklearn.metrics import explained_variance_score
import plotly.express as px
import plotly.graph_objects as go


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


'''
Class which implements a UANN for multi class classification
Initialize with:  
     input_dim: int = 13, --> number of input features
     n_hidden: list = [50], --> list of number of output features for each hidden layer (longer list = more complex model)
     dropout_rate: float =0.01, --> dropout rate to be used
     tau:float = 0.15, --> precision parameter, for more information see Gal et al. Dropout as Bayesian Approximation
     lambda_u: float = 0.2, --> factor to multiply uncertainty loss (see function combibed_loss)
     lambda_c: float = 0.4, --> factor to multiply classificatuion loss (see function combibed_loss)
     num_samples: int = 10000, --> Number of forward passes conducted with MC dropout
     file_path: str = 'Model' --> path where models and model definition are stored
'''
class UQNN(nn.Module):
    def __init__(self, 
                 input_dim: int = 13, 
                 n_hidden: list = [50], 
                 dropout_rate: float =0.01, 
                 tau:float = 0.15,
                 lambda_u: float = 0.2, 
                 lambda_r: float = 0.4, 
                num_samples: int = 10000, 
                file_path: str = 'Model'):
        super(UQNN, self).__init__()

        #Define model structure
        layers = []
        prev_dim = input_dim
        self.dropout = dropout_rate
        self.tau = tau
        
        self.fc_1 = nn.Linear(input_dim, 50)
        
        
        layers.append(nn.Dropout(dropout_rate))
        for h in n_hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        
        self.feature_extractor = nn.Sequential(*layers)
        
        self.regression_head =  nn.Sequential(nn.Linear(prev_dim, 80), 
                                              nn.ReLU(), 
                                              nn.Linear(80, 60),
                                              nn.ReLU(),
                                              nn.Linear(60, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(prev_dim, 80), 
                                              nn.ReLU(), 
                                              nn.Linear(80, 60),
                                              nn.ReLU(),
                                              nn.Linear(60, 1),
                                              nn.ReLU())
        
    
  

        #Parameters for monitoring training
        self.is_converged = False
        self.reg_losses = []
        self.combined_losses = []
        self.num_samples = num_samples

        #Parameters for training
        self.lambda_u = lambda_u
        self.lambda_r = lambda_r
                  
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
    Function which determines if regressor converged
    Automatically called during training
    Works based on variation in last losses
        --> value for variation may be adapted when mregressor does not converge or converges to fast 
    Minimum of 5 epochs necessary
        --> may be reduced if model converges quickly
    '''      
    def regressor_converged(self):
        if(len(self.reg_losses) < 5):
            return
        else:
            self.is_converged = True
        '''
        elif(len(self.combined_losses) > 0):
            self.is_converged = True
        else:
            min_loss = np.min(self.reg_losses[-20:])
            max_loss = np.max(self.reg_losses[-20:])
            variation = (max_loss - min_loss)/min_loss
            if(variation > 0.02):
                return
            else:
                self.is_converged = True
        '''

    def combined_loss(self, mu, label, sigma, uncertainty):
        reg_loss = F.mse_loss(mu, label) * self.lambda_r
        uncertainty_loss = F.mse_loss(sigma, uncertainty) * self.lambda_u
        return(uncertainty_loss + reg_loss).mean()
    
    def mc_predict(self, 
                   test_loader: DataLoader, 
                   tau: float = 1.0):
        self.feature_extractor.train()  # Keep dropout active for MC Dropout
        all_preds = []
        y_true_list = []
        all_features = []
        
        with torch.no_grad():
            for X_batch, y_batch in iter(test_loader):
                y_true_list.append(y_batch.numpy())
                preds = []
                features = []
                for _ in range(self.num_samples):
                    pred, feature = self.single_pass(X_batch)
                    preds.append(pred.item())
                    features.append(feature.cpu().detach().numpy())
                mean_features = np.mean(features, axis = 0)
                all_preds.append(preds)
                all_features.append(features)
        
        y_true = np.concatenate(y_true_list)
        
        Yt_hat = np.concatenate(all_preds)
        MC_pred = np.mean(Yt_hat, axis=0)
        v = np.var(Yt_hat, axis=0)  # Compute variance explicitly
        rmse = np.sqrt(np.mean((y_true - MC_pred) ** 2))
        mean_features = np.mean(features, axis = 0)
        
        # Fix log likelihood computation
        ll = -0.5 * np.log(2 * np.pi * (1/tau + v)) - 0.5 * ((y_true - MC_pred) ** 2) / (1/tau + v)
        test_ll = np.mean(ll)
        
        return rmse, test_ll, v, MC_pred, mean_features, all_preds

    def train_model(self, 
              train_loader,
              test_loader: DataLoader = None,
              num_epochs: int = 50):
        best_loss_reg = np.inf
        best_ll_combined = -np.inf
        best_loss_combined = np.inf
        reg_params = list(self.feature_extractor.parameters()) + list(self.regression_head.parameters())
        #uncertainty_params = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters()) + list(self.fc4.parameters()) + list(self.uncertainty_head.parameters())
        #uncertainty_params =  list(self.fc4.parameters()) + list(self.uncertainty_head.parameters())
        uncertainty_params =  list(self.feature_extractor.parameters())+ list(self.uncertainty_head.parameters()) + list(self.regression_head.parameters())

        self.reg_optim = optim.Adam(reg_params, lr=0.001, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-7)
        self.uncertainty_optim = optim.Adam(uncertainty_params, lr=0.001)#, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-7)
        for epoch in range(num_epochs):
            train_loss = 0.0
            test_loss = 0.0
            ll = 0.0
            if(self.is_converged == False):
                self.regressor_converged()
            if(self.is_converged == False):
                print(f'epoch {epoch} - Regressor not yet converged')
                for x, y in iter(train_loader):
                    pred, _ = self.single_pass(x)
                    target = y.view(-1, 1).float()
                    loss = F.mse_loss(pred, target)
                    train_loss += loss.item()
                    self.reg_optim.zero_grad()
                    loss.backward()
                    self.reg_optim.step()
                if test_loader is  not None:
                    for x, y in iter(test_loader):
                        with torch.no_grad():
                            pred, _ = self.single_pass(x)
                            target = y.view(-1, 1).float()
                            loss = F.binary_cross_entropy_with_logits(pred, target)
                            test_loss += loss.item()
                    test_loss /= len(test_loader)
                else:
                    test_loss = train_loss / len(train_loader)
                train_loss /= len(train_loader)
                
                if(test_loss < best_loss_reg):
                    torch.save(self.state_dict(), f'{self.file_path}/reg_epoch_{epoch}_loss_{int(test_loss * 100)}.pth')
                    best_loss_reg = test_loss
                print(f'Epoch {epoch}\nLL: {int(ll * 100)}\nValidation Loss: {test_loss}\nTrain Loss: {train_loss}\n\n')
                self.reg_losses.append(test_loss)
            else:
                print(f'epoch {epoch} - Regressor converged')
                
                for x, y in iter(train_loader):
                    preds = []
                    features = []
                    for _ in range(self.num_samples):
                        pred, feature = self.single_pass(x)
                        preds.append(pred)
                        features.append(feature)
                    preds= torch.stack(preds)
                    features = torch.stack(features)
                    features = features.mean(dim = 0)
                    mu = preds.mean(dim = 0)
                    sigma = preds.std(dim = 0)
                    uncertainty = self.uncertainty_head(features)
                    target = y.view(-1, 1).float()
                    loss = self.combined_loss(mu, target, sigma, uncertainty)
                    #loss = F.mse_loss(sigma, uncertainty)
                    train_loss += loss.item()
                    self.uncertainty_optim.zero_grad()
                    loss.backward()
                    self.uncertainty_optim.step()
                train_loss /= len(train_loader)

                uncertainties = []
                sigmas = []
                mus = []
                ys = []
                if(test_loader is not None):
                    for x, y in iter(test_loader):
                        preds = []
                        features = []
                        ys.append(y.item())
                        with torch.no_grad():
                            for _ in range(self.num_samples):
                                pred, feature = self.single_pass(x)
                                preds.append(pred)
                                features.append(feature)
                            preds= torch.stack(preds)
                            features = torch.stack(features)
                            features = features.mean(dim = 0)
                            mu = preds.mean(dim = 0)
                            mus.append(mu.item())
                            sigma = preds.std(dim = 0)
                            uncertainty = self.uncertainty_head(features)
                            sigmas.extend(list(sigma.detach().numpy().reshape(-1)))
                            uncertainties.extend(list(uncertainty.detach().numpy().reshape(-1)))
                            target = y.view(-1, 1).float()
                            loss = self.combined_loss(mu, target, sigma, uncertainty)
                            #loss = F.mse_loss(sigma, uncertainty)
                            test_loss += loss.item()
                    test_loss /= len(test_loader)
                    ev_uncertainty = explained_variance_score(uncertainties, sigmas)
                    ev_reg = explained_variance_score(mus, ys)
                    variance = np.var(mus)
                    ys = np.array(ys)
                    mus = np.array(mus)
                    ll = -0.5 * np.log(2 * np.pi * (1/self.tau + variance)) - 0.5 * ((ys - mus) ** 2) / (1/self.tau + variance)
                    ll = np.mean(ll)
                else:
                    test_loss = train_loss
                    ll = 0
                    ev_reg = 0
                    ev_uncertainty = 0
                if(test_loss < best_loss_combined):
                    torch.save(self.state_dict(), f'{self.file_path}/combined_epoch_{epoch}_ev_reg_{int(ev_reg*100)}_ev_uncertainty_{int(ev_uncertainty*100)}.pth')
                    best_loss_combined = test_loss
                elif(ll > best_ll_combined):
                    best_ll_combined = ll
                    torch.save(self.state_dict(), f'{self.file_path}/combined_epoch_{epoch}_ev_reg_{int(ev_reg*100)}_ev_uncertainty_{int(ev_uncertainty*100)}.pth')
                print(f'Log Likelihood: {ll}\nExplained Variance Uncerainty: {ev_uncertainty}\nExplained Variance Regression: {ev_reg}\nValidation Loss: {test_loss}\nTrain Loss: {train_loss}\n\n') 
                self.combined_losses.append(test_loss)
    
        """
        Falls mc_dropout=True, wird MC-Dropout aktiviert und mehrere Vorhersagen gemacht.
        """
        pred, features = self.single_pass(x)
        uncertainty = self.uncertainty_head(features)
        return pred, uncertainty
        
    def single_pass(self, x):
        x = F.relu(self.feature_extractor(x))
        output = self.regression_head(x)
        return output, x
        
    def forward(self, x):
        x = F.relu(self.feature_extractor(x))
        pred = self.regression_head(x)
        un = self.uncertainty_head(x)
        return pred, un
    
    def get_uncertainty_metrics(self, 
                                x):
        """
        Berechnet Entropie und Mutual Information über mehrere Vorhersagen hinweg.
        
        Args:
            model: Das trainierte Modell mit aktivem Dropout.
            input_data: Die Eingabedaten (ein einzelnes Bild oder ein Batch).
            n_samples: Anzahl der Forward-Passes mit aktivem Dropout.
        
        Returns:
            mean_entropy: Erwartungswert der Entropie über alle Vorhersagen.
            mutual_information: Mutual Information (Informationsgewinn).
        """
        self.train()
    
        # Liste zum Speichern der Softmax-Wahrscheinlichkeiten
        preds = []
        features = []
        # Mehrere Forward-Passes mit aktivem Dropout
        for _ in range(self.num_samples):
            with torch.no_grad():
                pred, feature = self.single_pass(x)  # Vorhersage mit Dropout
                preds.append(pred.squeeze())  # Hinzufügen zur Liste
                features.append(feature.squeeze())
                
        mean_features = torch.stack(features)
        mean_features = mean_features.mean(dim = 0)
        # Umwandlung in Tensor (Shape: [n_samples, batch_size, num_classes])
        preds = torch.stack(preds)
        mean_preds = torch.mean(preds, axis = 0)
        sigma = preds.std(dim=0)
        
        return mean_preds, sigma, mean_features
    
    
    def test_performance(self, 
                         data_loader: DataLoader):
        preds = []
        classes = []
        sigmas = []
        uncertainties = []
        labels = []
        for x, y in iter(data_loader):
            pred, sigma, feature = self.get_uncertainty_metrics(x)
            preds.append(pred)    
            #classes.extend([c.item() for c in classes])
            labels.extend([elem.item() for elem in y])
            uncertainty = self.uncertainty_head(feature)
            uncertainties.extend([u.item() for u in uncertainty])
            sigmas.append(sigma)
            
        #if(plot):
        return    uncertainties, sigmas, preds, labels
    
    
