#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:14:15 2021

@author: Ali Taatian
"""

import torch 
import torch.nn as nn

# Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, batch_size = 10, 5, 1, 10

# Create dummy input and target tensors (data)
x = torch.randn(batch_size, n_in)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])


# Create a model
model = nn.Sequential(nn.Linear(n_in, n_h),
   nn.ReLU(),
   nn.Linear(n_h, n_out),
   nn.Sigmoid())


# Construct the loss function
criterion = torch.nn.MSELoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Gradient Descent
for epoch in range(50):
   # Forward pass: Compute predicted y by passing x to the model
   y_pred = model(x)

   # Compute and print loss
   loss = criterion(y_pred, y)
   print('epoch: ', epoch,' loss: ', loss.item())

   # Zero gradients, perform a backward pass, and update the weights.
   optimizer.zero_grad()

   # perform a backward pass (backpropagation)
   loss.backward()

   # Update the parameters
   optimizer.step()
   
##################
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

path = r"/Users/nooshinnejati/Downloads"

trainset = torchvision.datasets.CIFAR10(root = '/Users/nooshinnejati/Downloads/data', train = True, download = True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)

landmarks_frame = pd.read_csv('/Users/nooshinnejati/Downloads//face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
#landmarks = landmarks.astype('float').reshape(-1, 2)

################################  Linear Regression
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd


sns.set_style(style = 'whitegrid')
plt.rcParams["patch.force_edgecolor"] = True

##### creating synethsized regression data
m = 2 # slope
c = 3 # intercept
x = np.random.rand(256)

noise = np.random.randn(256) / 4

y = x * m + c + noise

df = pd.DataFrame()
df['x'] = x
df['y'] = y

#sns.lmplot(x ='x', y ='y', data = df)

###### Implementing linear regression

import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')

class LinearRegressionModel(nn.Module):
   def __init__(self, input_dim, output_dim):
      super(LinearRegressionModel, self).__init__()
      self.linear = nn.Linear(input_dim, output_dim)

   def forward(self, x):
      out = self.linear(x)
      return out

input_dim = x_train.shape[1]
output_dim = y_train.shape[1]

### defining the model
### defining the loss function
criterion = nn.MSELoss()
model = LinearRegressionModel(input_dim, output_dim)

[w, b] = model.parameters()
print([w,b])
 
def get_param_values():
   return w.data[0][0], b.data[0]

def plot_current_fit(title = ""):
    plt.figure(figsize = (12,4))
    plt.title(title)
    
    plt.scatter(x, y, s = 8)
    
#    w.data[0][0]=2  #### Ali
    w1 = w.data[0][0]
    
#    b.data[0]=3     #### Ali
    b1 = b.data[0]
    
    x1 = np.array([0., 1.])

    y1 = tf.math.multiply(x1,w1) + b1
    
    plt.plot(x1, y1, 'r', label = 'Current Fit ({:.3f}, {:.3f})'.format(w1, b1))
    plt.xlabel('x (input)')
    plt.ylabel('y (target)')
    plt.legend()
    plt.show()
    print([w1,b1])
    

plot_current_fit('Before training')



############################ Implementing CNN
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleCNN(torch.nn.Module):
   def __init__(self):
      super(SimpleCNN, self).__init__()
      #Input channels = 3, output channels = 18
      self.conv1 = torch.nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
      self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
      #4608 input features, 64 output features (see sizing flow below)
      self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
      #64 input features, 10 output features for our 10 defined classes
      self.fc2 = torch.nn.Linear(64, 10)


def forward(self, x):
   x = F.relu(self.conv1(x))
   x = self.pool(x)
   x = x.view(-1, 18 * 16 *16)
   x = F.relu(self.fc1(x))
   #Computes the second fully connected layer (activation applied later)
   #Size changes from (1, 64) to (1, 10)
   x = self.fc2(x)
   return(x)


############################ Implementing RNN

import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init

dtype = torch.FloatTensor
input_size, hidden_size, output_size = 7, 6, 1
epochs = 300
seq_length = 20
lr = 0.1
data_time_steps = np.linspace(2, 10, seq_length + 1)
data = np.sin(data_time_steps)
data.resize((seq_length + 1, 1))

x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad=False)

w1 = torch.FloatTensor(input_size, hidden_size).type(dtype)
init.normal(w1, 0.0, 0.4)
w1 = Variable(w1, requires_grad = True)
w2 = torch.FloatTensor(hidden_size, output_size).type(dtype)
init.normal(w2, 0.0, 0.3)
w2 = Variable(w2, requires_grad = True)

def forward(input, context_state, w1, w2):
   xh = torch.cat((input, context_state), 1)
   context_state = torch.tanh(xh.mm(w1))
   out = context_state.mm(w2)
   return (out, context_state)


for i in range(epochs):
   total_loss = 0
   context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad = True)
   for j in range(x.size(0)):
      input = x[j:(j+1)]
      target = y[j:(j+1)]
      (pred, context_state) = forward(input, context_state, w1, w2)
      loss = (pred - target).pow(2).sum()/2
      total_loss += loss
      loss.backward()
      w1.data -= lr * w1.grad.data
      w2.data -= lr * w2.grad.data
      w1.grad.data.zero_()
      w2.grad.data.zero_()
      context_state = Variable(context_state.data)
   if i % 10 == 0:
      print("Epoch: {} loss {}".format(i, total_loss.data))


context_state = Variable(torch.zeros((1, hidden_size)).type(dtype), requires_grad = False)
predictions = []

for i in range(x.size(0)):
   input = x[i:i+1]
   (pred, context_state) = forward(input, context_state, w1, w2)
   context_state = context_state
   predictions.append(pred.data.numpy().ravel()[0])


pl.scatter(data_time_steps[:-1], x.data.numpy(), s = 90, label = "Actual")
pl.scatter(data_time_steps[1:], predictions, label = "Predicted")
pl.legend()
pl.show()

#######################       Datasets

#dset.MNIST(root, train = TRUE, transform = NONE, target_transform = None, download = FALSE)

################# COCO
import torchvision
#from torchvision import datasets
import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = "dir where images are", annFile = '/Users/nooshinnejati/Downloads/AI_Dataset.json',transform = transforms.ToTensor())
print("Number of samples: ", len(cap))
print(target)






