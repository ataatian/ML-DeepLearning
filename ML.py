#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 12:26:13 2021

@author: AliTaatian
"""

########## House cleaning
import os
os.system('clear')

from IPython import get_ipython
get_ipython().magic('reset -sf')
#############

import pandas as pd 

################### Loading pima-indians-diabetes.csv
from numpy import loadtxt
path = r"/Users/nooshinnejati/Downloads/pima-indians-diabetes.csv"


datapath= open(path, 'r')
data = loadtxt(datapath, delimiter=",")
print(data.shape)
print(type(data))
print(data[:3])
df = pd.DataFrame(data)
print(df[:3])

#################### Loading iris.csv
import csv
import numpy as np

path = r"/Users/nooshinnejati/Downloads/iris.csv"


with open(path,'r') as f:
   reader = csv.reader(f,delimiter = ',')
   headers = next(reader)
   data = list(reader)
   data = np.array(data).astype(float)
   
print(headers)
print(data.shape)
print(data[:3])
############

data = pd.read_csv(path)
print(data.dtypes)
#######
from matplotlib import pyplot

df.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()

#### Scaling
from sklearn import preprocessing

data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_rescaled = data_scaler.fit_transform(df)

print(data_rescaled[:3])

########## Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

path = r"/Users/nooshinnejati/Downloads/pima-indians-diabetes.csv"

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv(path, names=names)

array = df.values

X = array[:,0:8]
Y = array[:,8]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,Y)

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Number of Features:", fit.n_features_)
print("Selected Features:", fit.ranking_)
print("Feature Ranking: ", fit.support_)


print(X[:3])
print(Y[:3])
print(array[:3])

##### Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
fit = pca.fit(X)
print("Explained Variance:", fit.explained_variance_ratio_) 
print(fit.components_)

print(fit)

###### Feature Importance
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)


#####################################################
###################### Classification
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
print(type(data))
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

print(features)

## 1) Organizing data into training & testing sets
from sklearn.model_selection import train_test_split

train, test, train_labels, test_labels = train_test_split(
   features,labels,test_size = 0.40, random_state = 42
)

## 2) Model evaluation (Naive Bayes)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)

## 3) Finding accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,preds))
## Classification Evaluation Metrics
## 3-1) Confusion Matrix

from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_labels,preds))

###############################
######### SVM

######## example 1
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.50)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer');
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
plt.plot([0.6], [2.1], 'x', color='black', markeredgewidth=4, markersize=12)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
   plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5); 

from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

def decision_function(model, ax=None, plot_support=True):
    if ax is None:
       ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
   
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors='k',\
               levels=[-1, 0, 1], alpha=0.5,\
                   linestyles=['--', '-', '--'])
    if plot_support:
       ax.scatter(model.support_vectors_[:, 0],
          model.support_vectors_[:, 1],
          s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
decision_function(model);

########## example 2
from sklearn import svm, datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
   np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

C = 1.0

svc_classifier = svm.SVC(kernel='linear', C=C).fit(X, y)
Z = svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with linear kernel')


###################################
################## Decision Trees
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(r"/Users/nooshinnejati/Downloads/pima-indians-diabetes.csv", header=None, names=col_names)
pima.head()
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


############ Visualizing Decision Tree
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
      filled=True, rounded=True,
      special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

#(graph,)=pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_png('Pima_diabetes_Tree.png')
Image(graph.create_png())

############ Naive Bayes
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs
X, y = make_blobs(300, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer');

print(X[0:4,:])
print(y[0:4])
## GaussianNB model: training (fitting) the model
from sklearn.naive_bayes import GaussianNB
model_GNB = GaussianNB()
model_GNB.fit(X, y);

## creating some new data
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
print(Xnew)

# predicting the class (classifying)
ynew = model_GNB.predict(Xnew)
print(ynew)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=70, cmap='summer', alpha=0.1)
plt.axis(lim);


yprob = model_GNB.predict_proba(Xnew)
print(yprob.round(3))
print([yprob.round(3)[:,1]])
P=yprob.round(3)[:,1]

L=pd.DataFrame(list(zip(ynew,P)))
print(L)


################# Random Forest
import csv

path = r"/Users/nooshinnejati/Downloads/iris.csv"


dataset = pd.read_csv(path)

dataset.head()
   
print(headers)


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print(X)

###### Splitting the data to Training and Test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print(X_train)

##### Training the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)


### Predicting
y_pred = classifier.predict(X_test)
print(y_pred)


###### Performance Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

L=pd.DataFrame(list(zip(y_test,y_pred)))
print(L)

#################################################################
########################  Clustering

############ K-means
from sklearn.cluster import KMeans

##### example 1
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=20);
plt.show()
print(X)
del y_true ##no use

## Model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

## visualization
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='summer')
centers = kmeans.cluster_centers_
print(centers)
plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9);
plt.show()



##### example 2: Digit recognition
from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape
print(digits.data)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(digits.data)
kmeans.cluster_centers_.shape
print(clusters)

## The centers (representatives)
centers = kmeans.cluster_centers_.reshape(10, 8, 8)

## Illustration
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
for axi, center in zip(ax.flat, centers):
   axi.set(xticks=[], yticks=[])
   axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

## evaluation
from scipy.stats import mode
labels = np.zeros_like(clusters)
print(labels)
for i in range(10):
   mask = (clusters == i)
   labels[mask] = mode(digits.target[mask])[0]

from sklearn.metrics import accuracy_score
print(digits.target) ### The true labels
accuracy_score(digits.target, labels)


############  Mean Shift
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.datasets.samples_generator import make_blobs
centers = [[3,3,3],[4,5,5],[3,10,10]] #the true cluster centers (3D)
#centers = [[3,3],[4,5],[3,10]]

X, _ = make_blobs(n_samples = 700, centers = centers, cluster_std = 0.5)
plt.scatter(X[:,0],X[:,1])
plt.show()
print(X)


ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_  #the predicted cluster centers
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)
colors = 10*['r.','g.','b.','c.','k.','y.','m.']
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 3)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
    marker=".",color='k', s=20, linewidths = 5, zorder=10)
plt.show()


##########################  Agglomerative Hierarchical Clustering


####### example 1
X = np.array([[7,8],[12,20],[17,19],[26,15],[32,37],[87,75],[73,85], [62,80],[73,60],[87,96],])
labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
   plt.annotate(label,xy=(x, y), xytext=(-3, 3),textcoords='offset points', ha='right', va='bottom')
plt.show()

## dendrograms
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
linked = linkage(X, 'single')
labelList = range(1, 11)
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top',labels=labelList, distance_sort='descending',show_leaf_counts=True)
plt.show()


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')

####### example 2
from pandas import read_csv
path = r"/Users/nooshinnejati/Downloads/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
data.shape
data.head()

patient_data = data.iloc[:, 3:5].values
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Patient Dendograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))


from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit_predict(patient_data)
plt.figure(figsize=(10, 7))
plt.scatter(patient_data[:,0], patient_data[:,1], c=cluster.labels_, cmap='rainbow')



#############  KNN as Classifier
import csv

path = r"/Users/nooshinnejati/Downloads/iris.csv"

dataset = pd.read_csv(path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
print(X_train[0:4,:])

## data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

##### training the model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)


### predicting
y_pred = classifier.predict(X_test)

### evaluation
L=pd.DataFrame(list(zip(y_test,y_pred)))
print(L)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

###################################################################
####################### ML pipelines

## example 1
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

path = r"/Users/nooshinnejati/Downloads/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
print(array)
X = array[:,0:8]
Y = array[:,8]


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

#### evaluation by cross-validation (20 times prediction and calculating the mean of the accuracies)
kfold = KFold(n_splits=20, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


## example 2
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)

kfold = KFold(n_splits=20, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


######################## Bagging Ensemble Algorithms

################ Bagged Decision Tree
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

path = r"/Users/nooshinnejati/Downloads/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
print(array)
X = array[:,0:8]
Y = array[:,8]


seed = 7
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()

num_trees = 150

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

################### Random Forest
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

path = r"/Users/nooshinnejati/Downloads/pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
array = data.values
print(array)
X = array[:,0:8]
Y = array[:,8]

seed = 7
kfold = KFold(n_splits=10, random_state=seed)

num_trees = 150
max_features = 5

model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

################# Extra Trees

from sklearn.ensemble import ExtraTreesClassifier

seed = 7
kfold = KFold(n_splits=10, random_state=seed)

num_trees = 150
max_features = 5

model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)

results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


###################### Boosting Ensemble Algorithms

##############  AdaBoost
from sklearn.ensemble import AdaBoostClassifier

seed = 5
kfold = KFold(n_splits=10, random_state=seed)

num_trees = 50

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


#############  Stochastic Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

seed = 5
kfold = KFold(n_splits=10, random_state=seed)

num_trees = 50

model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)

results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


##################### Voting Ensemble Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = KFold(n_splits=10, random_state=7)

estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())



###################### Grid Search Parameter Tuning
import numpy
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)


model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)

print(grid.best_score_)
print(grid.best_estimator_.alpha)



####################### Random Search Parameter Tuning
from scipy.stats import uniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

param_grid = {'alpha': uniform()}
model = Ridge()
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50,random_state=7)
random_search.fit(X, Y)

print(random_search.best_score_)
print(random_search.best_estimator_.alpha)





