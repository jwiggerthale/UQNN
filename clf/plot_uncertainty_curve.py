'''
This script plots the uncertainty accuracy curve for a classification model using: 
  ensembles
  MC dropout
  UQNN
'''

#Import libraries
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from typing import Union, Callable, Tuple, Optional, Text, Sequence, Dict
import numpy as np
import pandas as pd
from UANN_Classification import UANN_Classifier
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from ensemble_lenet import LeNet5
import scipy
import os


'''
Function which makes prediction and calculates uncertainty using ensmebles 
Call with: 
  model_list --> list of file paths for single models
  im --> image to be classified
Reurns: 
  mu --> mean prediction of models
  sigma --> std of predictions divided by mu
'''
def ensemble_fwd(model_list: list, im: torch.tensor):
    models = []
    for fp in model_list:
        model = LeNet5()
        model.load_state_dict(torch.load(fp))
        model.eval()
        models.append(model)
        
    outputs = []
    for model in models:
        output = model.forward(im)
        outputs.append(output)
    outputs = torch.stack(outputs)
    mu = outputs.mean(dim = 0)
    sigma = outputs.std(dim = 0)
    sigma /= mu
    return mu, torch.abs(sigma)

    
    mc_samples = np.asarray([list(model.forward(im).detach().cpu().numpy()) for model in models])

    # Bernoulli output distribution
    dist = scipy.stats.bernoulli(mc_samples.mean(axis=0))
    # Predictive mean calculation
    mean = dist.mean()
    uncertainty = dist.std()
    return mean, uncertainty
      

'''
Function which evaluates a model either on AUC or accuracy
Call with: 
  model --> model to be tested
  dataloader --> dataloader containing data for test
  model_type --> type of model to be tested (one of 'UQNN', 'MC' or 'ensemble')
  model_list --> list of file paths of ensemble models (if model_type == 'ensemble')
Retruns: 
  dictionary containing two DataFrames (one for accuracy and one for AUC)
    --> each DataFrame has column mean and fraction
    --> mean is AUC / accuracy of model for appropriiate fraction
'''
def evaluate(
      model: nn.Module, 
      dataloader: DataLoader, 
      model_type: str = 'UQNN',
      model_list: list = None
    ) -> Dict[Text, float]:
   
    # Containers used for caching performance evaluation
    y_true = list()
    labels = []
    y_pred = list()
    preds = []
    classes = []
    y_uncertainty = list()
    uncertainties = []

    #Define prediction function for model_type
    if model_type == 'UQNN':
        fwd = model.forward
    elif model_type == 'MC':
        fwd = model.mc_predict
    elif model_type == 'ensemble':
        fwd = lambda im: ensemble_fwd(model_list, im)

    #Get prediction and uncertainy for each sample
    for x, y in iter(dataloader):
      if isinstance(x, list):
          for im in x:
              pred, uncertainty = fwd(im)
              labels.extend([elem.int().item() for elem in y])
              np_pred = F.softmax(pred).cpu().detach().numpy()
              preds.extend([list(elem) for elem in np_pred])
              predictions = [elem.argmax().item() for elem in pred]
              classes.extend([elem.argmax().item() for elem in pred])
              uncertainties.extend([u[predictions[i]].item() for i, u in enumerate(uncertainty)])
      else:
        pred, uncertainty = fwd(x)
        labels.extend([elem.int().item() for elem in y])
        np_pred = F.softmax(pred).cpu().detach().numpy()
        preds.extend([list(elem) for elem in np_pred])
        predictions = [elem.argmax().item() for elem in pred]
        classes.extend([elem.argmax().item() for elem in pred])
        uncertainties.extend([u[predictions[i]].item() for i, u in enumerate(uncertainty)])
  
  
    # Use vectorized NumPy containers
    labels = np.array(labels)
    preds = np.array(preds)
    classes = np.array(classes)
    uncertainties = np.array(uncertainties)
    #y_true = np.concatenate(y_true).flatten()
    #y_pred = np.concatenate(y_pred).flatten()
    #y_uncertainty = np.concatenate(y_uncertainty).flatten()
    fractions = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
  
    return {
        'acc': _evaluate_metric(
            labels,
            classes,
            uncertainties,
            fractions,
            accuracy_score
        ), 

        'auc': _evaluate_metric(
            labels,
            preds,
            uncertainties,
            fractions,
            roc_auc_score, 
            metric = 'roc'
        )
    }


'''
Function which evaluates models predictions on a metric function
Called automatically from function evaluate 
Call with: 
  y_true --> labels
  y_pred --> predictions
  y_uncertainty --> uncertainties
  fractions --> list of fractions to be retained
  metric_fn --> sklearn.metrics.accuracy_score or sklearn.metrics.roc_auc_score
Returns: 
  DataFrames for performance
    --> each DataFrame has column mean and fraction
    --> mean is AUC / accuracy of model for appropriiate fraction
'''
def _evaluate_metric(
      y_true: np.ndarray,
      y_pred: np.ndarray,
      y_uncertainty: np.ndarray,
      fractions: Sequence[float],
      metric_fn: Callable[[np.ndarray, np.ndarray], float],
      metric: str = 'acc'
    ) -> pd.DataFrame:
    """Evaluate model predictive distribution on `metric_fn` at data retain
    `fractions`.
    
    Args:
      y_true: `numpy.ndarray`, the ground truth labels, with shape [N].
      y_pred: `numpy.ndarray`, the model predictions, with shape [N].
      y_uncertainty: `numpy.ndarray`, the model uncertainties,
        with shape [N].
      fractions: `iterable`, the percentages of data to retain for
        calculating `metric_fn`.
      metric_fn: `lambda(y_true, y_pred) -> float`, a metric
        function that provides a score given ground truths
        and predictions.
      name: (optional) `str`, the name of the method.
    
    Returns:
      A `pandas.DataFrame` with columns ["retained_data", "mean", "std"],
      that summarizes the scores at different data retained fractions.
    """
    
    N = y_true.shape[0]
    
    # Sorts indexes by ascending uncertainty
    I_uncertainties = np.argsort(y_uncertainty)
    
    # Score containers
    mean = np.empty_like(fractions)
    # TODO(filangel): do bootstrap sampling and estimate standard error
    std = np.zeros_like(fractions)
    
    for i, frac in enumerate(fractions):
      # Keep only the %-frac of lowest uncertainties
      I = np.zeros(N, dtype=bool)
      I[I_uncertainties[:int(N * frac)]] = True
      I = np.array(I)
      if metric == 'roc':
          mean[i] = metric_fn(y_true[I], 
                                y_pred[I], 
                                multi_class="ovr",
                                average="macro")
      else:
          mean[i] = metric_fn(y_true[I], y_pred[I])
    
    # Store
    df = pd.DataFrame(dict(retained_data=fractions, mean=mean, std=std))
    
    return df


'''
Our mnist datset 
--> for more information see train_UQNN.py
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


'''
Similar as mnist_dataset but returns list of images roteted by certain angles
Call with: 
  df --> DataFrame containing Images and labels
  angles --> list of rotation angles to be used for testing
'''
class rotation_mnist_dataset(Dataset):
    def __init__(self, df: pd.DataFrame, angles: list):
        self.data = df
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081]),
        ])
        self.rotation_angles = angles # 0 bis 95 Grad in 5-Grad-Schritten

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

#Get data and create data loader
train = pd.read_csv('/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_train.csv').drop('Unnamed: 0', axis = 1)
test = pd.read_csv('/t1/erlangen/users/jwiggerthale/TestDatasets/MNIST/mnist_test.csv').drop('Unnamed: 0', axis = 1)
train_set = rotation_mnist_dataset(train, angles = [20])
test_set = rotation_mnist_dataset(test, angles = [20])
train_loader = DataLoader(train_set, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 16, shuffle = False)


#Define parameters for UQNN model
lambda_u = 0.1
lambda_c = 1.2
num_samples  = 100
num_classes = 10

model = UANN_Classifier(num_classes = num_classes,
                 lambda_u = lambda_u, 
                 lambda_c = lambda_c, 
                 num_samples = num_samples,
                 file_path = './clf')

model.load_state_dict(torch.load('./Clf_final_pretrained/model_pretrained.pth'))
model.eval() 

#Evaluate UQNN
results = evaluate(model, train_loader)


#Evaluate MC dropout
model.train() 
results_mc = evaluate(model, train_loader, model_type = 'MC')

#Evaluate ensemble
fp = './ensemble_models'
model_list = [f'{fp}/{f}' for f in os.listdir(fp) if ('39' in f and f.endswith('pth'))]
results_ensemble = evaluate(None, train_loader, model_type = 'ensemble', model_list = model_list)



#Get metrics and plot
acc = results['acc']
auc = results['auc']
acc_mc = results_mc['acc']
auc_mc = results_mc['auc']
acc_ensemble = results_ensemble['acc']
auc_ensemble = results_ensemble['auc']

fig, ax = plt.subplots(figsize =(12,12))
ax.plot(acc['retained_data'], acc['mean'], color = 'red', label = 'UQNN')
ax.plot(acc_mc['retained_data'], acc_mc['mean'], color = 'purple', label = 'MC')
ax.plot(acc_ensemble['retained_data'], acc_ensemble['mean'], color = 'blue', label = 'Ensemble')
ax.set_xlabel('Retained Data')
ax.set_xlabel('Accuracy')
fig.suptitle('Accuracy of Different Methods when Classifying Certain Share of Data')
fig.legend()









