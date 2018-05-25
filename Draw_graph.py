# -*- coding: utf-8 -*-
"""
Created on Tue May 22 23:43:45 2018

@author: kirby
"""

import pandas as pd
import numpy as np

from sklearn import svm, naive_bayes, tree, ensemble, neighbors, linear_model
from sklearn import neural_network
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

input_df = pd.read_csv("./train.csv")
        
# Last column is Class, so this needs to be removed from the data
raw_data = np.delete(input_df.values, 0, 1)

# Keep class col as the target       
input_target = input_df.values[:,0]

input_sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
input_data = input_sel.fit_transform(raw_data)

test_data = pd.read_csv("./test.csv")
raw_test = np.delete(test_data.values, 0, 1)
X_test = input_sel.transform(raw_test)
'''
#plot for C in SVM
cs = np.arange(1,25,2)
means=[]
for c in cs:
    
    clf = SVC(C=c,gamma=0.01)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(cs,means)
plt.xlabel('C')
plt.ylabel('score')
plt.title("C against score")
plt.show()

#plot for gamma in SVM
gammas = [1,0.1,0.01,0.001,0.0001,0.00001]
means=[]
for gamma in gammas:
    
    clf = SVC(C=20,gamma=gamma)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
print(gammas)
plt.plot(gammas,means)
plt.xlabel('gamma')
plt.ylabel('score')
plt.title("gamma against score")
plt.show()

#plot for kernel in SVM
kernels = ['linear', 'poly', 'rbf']
means=[]
for kernel in kernels:
    
    clf = SVC(kernel=kernel)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())

plt.plot(kernels,means)
plt.xlabel('kernel')
plt.ylabel('score')
plt.title("kernel against score")
plt.show()

#plot for max_depth in randomforest
depths = np.arange(1,60,2)
means=[]
for depth in depths:
    clf = ensemble.RandomForestClassifier(n_estimators=80,max_depth=depth)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(depths,means)
plt.xlabel('depth')
plt.ylabel('score')
plt.title("depth against score")
plt.show()

#plot for number of estimator in randomforest
estimators = np.arange(1,100,5)
means=[]
for e in estimators:
    clf = ensemble.RandomForestClassifier(n_estimators=e)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(estimators,means)
plt.xlabel('estimator')
plt.ylabel('score')
plt.title("estimator against score")
plt.show()

#plot for learning rate in gradient boosting
lrs = [0.2,0.15,0.1,0.05,0.025,0.01,0.005,0.001]
means=[]
for lr in lrs:
    clf = ensemble.GradientBoostingClassifier(learning_rate=lr)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(lrs,means)
plt.xlabel('learning rate')
plt.ylabel('score')
plt.title("learning rate against score")
plt.show()

#plot for estimator in gradient boosting
estimators = np.arange(1,100,10)
means=[]
for e in estimators:
    clf = ensemble.GradientBoostingClassifier(n_estimators=e)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(estimators,means)
plt.xlabel('estimator')
plt.ylabel('score')
plt.title("estimator against score")
plt.show()

#plot for maximum features in gradient boosting
features = np.arange(1,30,2)
means=[]
for f in features:
    clf = ensemble.GradientBoostingClassifier(max_features=f)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(features,means)
plt.xlabel('maximum feature')
plt.ylabel('score')
plt.title("maximum feature against score")
plt.show()

#plot for maximum depth in gradient boosting
depths = np.arange(1,30,2)
means=[]
for d in depths:
    clf = ensemble.GradientBoostingClassifier(max_depth=d)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(depths,means)
plt.xlabel('maximum depth')
plt.ylabel('score')
plt.title("maximum depth against score")
plt.show()

#plot for solver in MLP
solvers = ['adam', 'lbfgs', 'sgd']
means=[]
for solver in solvers:
    clf = neural_network.MLPClassifier(solver=solver)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(solvers,means)
plt.xlabel('solver')
plt.ylabel('score')
plt.title("solver against score")
plt.show()

#plot for learning rate in MLP
lrs = [0.2,0.1,0.05,0.025,0.01,0.005,0.001]
means=[]
for lr in lrs:
    clf = neural_network.MLPClassifier(learning_rate_init=lr,max_iter=1250)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(lrs,means)
plt.xlabel('learning rate')
plt.ylabel('score')
plt.title("learning rate against score")
plt.show()

#plot for learning rate in MLP
alphas = [0.025,0.01,0.005,0.001,0.0005,0.0001]
means=[]
for alpha in alphas:
    clf = neural_network.MLPClassifier(learning_rate_init=0.01,alpha=alpha)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(alphas,means)
plt.xlabel('alpha')
plt.ylabel('score')
plt.title("alpha against score")
plt.show()

#plot for hidden layer size in MLP
nodes = range(1,102,5)
means=[]
node1 = 0
node2 = 0
best = 0
for node in nodes:
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(node),learning_rate_init=0.01,alpha=0.001)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    if scores.mean() > best:
        best = scores.mean()
        node1 = node
    means.append(scores.mean())
print(node1)
plt.plot(nodes,means,label='1 layer')
plt.xlabel('hidden layer size')
plt.ylabel('score')
plt.title("Hidden layer size against score")
nodes = range(1,102,5)
means=[]
for node in nodes:
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(node1,node),learning_rate_init=0.01,alpha=0.001)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    if scores.mean() > best:
        best = scores.mean()
        node2 = node
    means.append(scores.mean())
plt.plot(nodes,means,label='2 layer')

plt.legend(loc='best')
plt.show()
'''
#plot for comparing data preprocessing in gradient boosting
depths = np.arange(1,30,2)
means=[]
for d in depths:
    clf = ensemble.GradientBoostingClassifier(max_depth=d)
    scores = cross_val_score(clf, input_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(depths,means,label='preprocessed')

depths = np.arange(1,30,2)
means=[]
for d in depths:
    clf = ensemble.GradientBoostingClassifier(max_depth=d)
    scores = cross_val_score(clf, raw_data, input_target, cv=5,scoring='roc_auc')
    means.append( scores.mean())
plt.plot(depths,means,label='not preprocessed')
plt.xlabel('maximum depth')
plt.ylabel('score')
plt.title("maximum depth against score")
plt.legend()
plt.show()