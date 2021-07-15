#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 16:43:19 2021

@author: Ali Taatian

"""

########## House cleaning
import os
os.system('clear')

from IPython import get_ipython
get_ipython().magic('reset -sf')
#############

######### A sample classification model

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd


###### 1) Data preparation
path = r"/Users/nooshinnejati/Downloads/Churn_Modelling.csv"

dataset = pd.read_csv(path)
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values
X
print(X)
print(type(X))
Y
y=Y

### encoding the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder() 
X[:,1] = labelencoder_X_1.fit_transform(X[:,1]) 
labelencoder_X_2 = LabelEncoder() 
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
X

##### Creating dummies for categorical variables
from sklearn.compose import ColumnTransformer 
onehotencoder = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
X = X[:, 1:]
X

#### Splitting the dataset into the Training set and the Test Set (train- to- test split ratio is 80:20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)
X_train
X_test

# Importing the Keras libraries and packages 

##### 2) Chosing the best (ANN) model

#import keras 
#from keras.models import Sequential 
from tensorflow.keras.models import Sequential 

#from keras.layers import Dense, Activation 
from tensorflow.keras.layers import Dense

#Initializing Neural Network
classifier = Sequential()

# layer 1: # of input variables:11
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# layer 2
classifier.add(Dense(units = 16, kernel_initializer = 'uniform',  activation = 'relu'))

# layer 3 (output layer)
classifier.add(Dense(units = 1, kernel_initializer = 'uniform',  activation = 'sigmoid'))

classifier.summary()

###### 3) Compiling the Neural Network (choosing loss function and optimizer)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

###### 4) Training the model (model fit)
classifier.fit(X_train, y_train, batch_size = 10, epochs = 50)


###### 5) Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred > 0.5)

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 500, 1, 40, 3, 50000, 2, 1, 1, 40000]])))
new_prediction = (new_prediction > 0.5)
print(new_prediction)


####### 6) Evaluating the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

##########################################################################
################################# The Forward Propagation Algorithm

import numpy as np
input_data = np.array([-1, 2])
weights = {
   'node_0': np.array([3, 3]),
   'node_1': np.array([1, 5]),
   'output': np.array([2, -1])
}
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = np.tanh(node_0_input)
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = np.tanh(node_1_input)
hidden_layer_output = np.array(node_0_output, node_1_output)
output =(hidden_layer_output * weights['output']).sum()
print(output)

def relu(input):
   '''Define your relu activation function here'''
   # Calculate the value for the output of the relu function: output
   output = max(input,0)
      # Return the value just calculated
   return(output)
# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()
print(model_output)# Print model output

##########
# Define predict_with_network()
def predict_with_network(input_data_row, weights):
   # Calculate node 0 value
   node_0_input = (input_data_row * weights['node_0']).sum()
   node_0_output = relu(node_0_input)
   
   # Calculate node 1 value
   node_1_input = (input_data_row * weights['node_1']).sum()
   node_1_output = relu(node_1_input)
   
   # Put node values into array: hidden_layer_outputs
   hidden_layer_outputs = np.array([node_0_output, node_1_output])
   
   # Calculate model output
   input_to_final_layer = (hidden_layer_outputs*weights['output']).sum()
   model_output = relu(input_to_final_layer)
# Return model output
   return(model_output)

# Create empty list to store prediction results
results = []
for input_data_row in input_data:
   # Append prediction to results
   results.append(predict_with_network(input_data_row, weights))
print(results)# Print results


#########################################
import numpy as np
input_data = np.array([3, 5])
weights = {
   'node_0_0': np.array([2, 4]),
   'node_0_1': np.array([4, -5]),
   'node_1_0': np.array([-1, 1]),
   'node_1_1': np.array([2, 2]),
   'output': np.array([2, 7])
}

def relu(input):
   '''Define your relu activation function here'''
   # Calculate the value for the output of the relu function: output
   output = max(input,0)
      # Return the value just calculated
   return(output)

def predict_with_network(input_data):
   # Calculate node 0 in the first hidden layer
   node_0_0_input = (input_data * weights['node_0_0']).sum()
   node_0_0_output = relu(node_0_0_input)
   
   # Calculate node 1 in the first hidden layer
   node_0_1_input = (input_data*weights['node_0_1']).sum()
   node_0_1_output = relu(node_0_1_input)
   
   # Put node values into array: hidden_0_outputs
   hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
   
   # Calculate node 0 in the second hidden layer
   node_1_0_input = (hidden_0_outputs*weights['node_1_0']).sum()
   node_1_0_output = relu(node_1_0_input)
   
   # Calculate node 1 in the second hidden layer
   node_1_1_input = (hidden_0_outputs*weights['node_1_1']).sum()
   node_1_1_output = relu(node_1_1_input)
   
   # Put node values into array: hidden_1_outputs
   hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])
   
   # Calculate model output: model_output
   model_output = (hidden_1_outputs*weights['output']).sum()
      # Return model_output
   return(model_output)
output = predict_with_network(input_data)
print(output)


#####################################################

##### Sequential models

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout 
from tensorflow.keras.layers import Dense 

model = Sequential()  

model.add(Dense(12, activation = 'relu', input_shape = (784,))) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation = 'relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation = 'softmax'))

model.summary()
#########################
from tensorflow.keras import backend as k

k.backend()
k.epsilon() 
k.image_data_format() 
k.floatx() 

##########
model = Sequential() 

myLayer=Dense(32, input_shape=(16,), kernel_initializer = 'he_uniform', kernel_regularizer = None, kernel_constraint = 'MaxNorm', activation = 'relu')
model.add(myLayer) 
model.add(Dense(16, activation = 'relu')) 
model.add(Dense(8))

myLayer.get_weights() 
myLayer.get_config()
myLayer.input_shape
myLayer.input

#################################################################
#################################### Creating customized layer
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Layer

class MyCustomLayer(Layer):
    def __init__(self, output_dim, **kwargs):    
       self.output_dim = output_dim 
       super(MyCustomLayer, self).__init__(**kwargs)
       
    def build(self, input_shape): 
       self.kernel = self.add_weight(name = 'kernel', 
          shape = (input_shape[1], self.output_dim), 
          initializer = 'normal', trainable = True) 
       super(MyCustomLayer, self).build(input_shape)

    def call(self, input_data): 
       return K.dot(input_data, self.kernel)

    def compute_output_shape(self, input_shape): return (input_shape[0], self.output_dim)


######### Main
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

model = Sequential() 
model.add(MyCustomLayer(32, input_shape = (16,))) 
model.add(Dense(8, activation = 'softmax')) 
model.summary()


################## Functional Models
from tensorflow.keras.layers import Input
data = Input(shape=(2,3))
print(data)

from tensorflow.keras.layers import Dense
layer = Dense(2)(data) 
print(layer) 

from tensorflow.keras.models import Model
model = Model(inputs = data, outputs = layer)
model.summary() 

###################################################################
import numpy as np 

x_train = np.random.random((100,4,8)) 
y_train = np.random.random((100,10))

x_val = np.random.random((100,4,8)) 
y_val = np.random.random((100,10))

from tensorflow.keras.models import Sequential 
model = Sequential()

from tensorflow.keras.layers import LSTM, Dense 

# add a sequence of vectors of dimension 16 
model.add(LSTM(16, return_sequences = True)) 
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 32, epochs = 5, validation_data = (x_val, y_val))
###############
######################## A sample project (with Perceptron)

from tensorflow.keras.datasets import mnist 
from tensorflow.keras.optimizers import RMSprop 
from tensorflow.keras.utils import to_categorical 


(x_train, y_train), (x_test, y_test) = mnist.load_data()


### preparing the data
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784) 
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 

y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10)

model = Sequential() 
model.add(Dense(512, activation = 'relu', input_shape = (784,))) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2)) 
model.add(Dense(10, activation = 'softmax'))


model.compile(loss = 'categorical_crossentropy',     
   optimizer = RMSprop(), 
   metrics = ['accuracy'])


history = model.fit(
   x_train, y_train, 
   batch_size = 128, 
   epochs = 20, 
   verbose = 1, 
   validation_data = (x_test, y_test)
)

#### evaluation
score = model.evaluate(x_test, y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

#### prediction
pred= model.predict(
   x_test, 
   batch_size = None, 
   verbose = 0, 
   steps = None, 
   callbacks = None, 
   max_queue_size = 10, 
   workers = 1, 
   use_multiprocessing = False
)

##### checking the first five images
pred = np.argmax(pred, axis = 1)[:5] 

label = np.argmax(y_test,axis = 1)[:5] 
print(pred) 
print(label)

#########################  A sample project (with CNN)
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = 28, 28 

if K.image_data_format() == 'channels_first': 
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) 
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
   input_shape = (1, img_rows, img_cols) 
else: 
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
   input_shape = (img_rows, img_cols, 1) 
   
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 

y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10)


model = Sequential() 
model.add(Conv2D(32, kernel_size = (3, 3),  activation = 'relu', input_shape = input_shape)) 
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(128, activation = 'relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(10, activation = 'softmax'))


model.compile(loss = categorical_crossentropy, optimizer = Adadelta(), metrics = ['accuracy'])


model.fit(
   x_train, y_train, 
   batch_size = 128, 
   epochs = 12, 
   verbose = 1, 
   validation_data = (x_test, y_test)
)

#### evaluation
score = model.evaluate(x_test, y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

pred = model.predict(x_test) 


pred = np.argmax(pred, axis = 1)[:5] 
label = np.argmax(y_test,axis = 1)[:5] 

print(pred) 
print(label)


################################# Regression prediction
from tensorflow.keras.datasets import boston_housing 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import RMSprop 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn import preprocessing 
from sklearn.preprocessing import scale

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

#### normalizing the data
x_train_scaled = preprocessing.scale(x_train) 
scaler = preprocessing.StandardScaler().fit(x_train) 
x_test_scaled = scaler.transform(x_test)


model = Sequential() 
model.add(Dense(64, kernel_initializer = 'normal', activation = 'relu',input_shape = (13,))) 
model.add(Dense(64, activation = 'relu')) 
model.add(Dense(1))


model.compile(
   loss = 'mse', 
   optimizer = RMSprop(), 
   metrics = ['mean_absolute_error']
)


history = model.fit(
   x_train_scaled, y_train,    
   batch_size=128, 
   epochs = 500, 
   verbose = 1, 
   validation_split = 0.2, 
   callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20)]
)

score = model.evaluate(x_test_scaled, y_test, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

prediction = model.predict(x_test_scaled) 
print(prediction.flatten()) 
print(y_test)

################################ Time Series Prediction using LSTM RNN
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Embedding 
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000)


#### preparing the data
x_train = sequence.pad_sequences(x_train, maxlen=80) 
x_test = sequence.pad_sequences(x_test, maxlen=80)


model = Sequential() 
model.add(Embedding(2000, 128)) 
model.add(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2)) 
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(
   x_train, y_train, 
   batch_size = 32, 
   epochs = 15, 
   validation_data = (x_test, y_test)
)

score, acc = model.evaluate(x_test, y_test, batch_size = 32) 
   
print('Test score:', score) 
print('Test accuracy:', acc)


##############################  Pre-trained models

import numpy as np 

from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet 

#Load the VGG model 
vgg_model = vgg16.VGG16(weights = 'imagenet') 

#Load the Inception_V3 model 
inception_model = inception_v3.InceptionV3(weights = 'imagenet') 

#Load the ResNet50 model 
resnet_model = resnet50.ResNet50(weights = 'imagenet') 

#Load the MobileNet model 
mobilenet_model = mobilenet.MobileNet(weights = 'imagenet')


#######################  Real Time Prediction using ResNet Model (image classification problem)


import PIL 
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications.imagenet_utils import decode_predictions 
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow.keras.applications.resnet50 import ResNet50 

path = r"/Users/nooshinnejati/Downloads/banana.jpg"
path = r"/Users/nooshinnejati/Downloads/fig.jpg"

## load an image in PIL format 
original = load_img(path, target_size = (224, 224)) 
print('PIL image size',original.size)
plt.imshow(original) 
plt.show()

#convert the PIL image to a numpy array
numpy_image = img_to_array(original) 

plt.imshow(np.uint8(numpy_image)) 
print('numpy array size',numpy_image.shape)

# Convert the image / images into batch format 
image_batch = np.expand_dims(numpy_image, axis = 0) 

print('image batch size', image_batch.shape) 

processed_image = resnet50.preprocess_input(image_batch.copy())

# create resnet model 
resnet_model = resnet50.ResNet50(weights = 'imagenet') 

# get the predicted probabilities for each class 
predictions = resnet_model.predict(processed_image)

# convert the probabilities to class labels 
label = decode_predictions(predictions)
print(label)

####################
from theano import *
a = tensor.dscalar()
b = tensor.dscalar()
c = a + b
f = theano.function([a,b], c)
theano.printing.pydotprint(f, outfile="/Users/nooshinnejati/Downloads/scalar_addition.png", var_with_name_simple=True)
