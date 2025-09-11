from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import sys
import matplotlib

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import accuracy_score #works
from matplotlib.ticker import FormatStrFormatter
import statsmodels.api as sm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import statsmodels.api as sm
rng = np.random.default_rng()
from scipy.stats import norm
#Adam Kainikara
#This code is for
#CHAPTER 5 QUESTION 6
#THIS IS PROBLEM 8
#OF HOMEWORK 2 FOR STANFORD SUMMER SESSION STATS 202

def data_loader(fname):
    num_data_a = loadtxt(fname,skiprows=1, usecols=(2,3), delimiter=',')
    defa_a = loadtxt(fname, skiprows=1, usecols=(0,1), delimiter=',', dtype=str)
    ydefault = [1 if x == "Yes" else 0 for x in defa_a[:,0]]
    ystudent = [1 if x == "Yes" else 0 for x in defa_a[:,1]]

    default_a = transpose(array((ydefault, ystudent)))
    #print(default_a)
    return num_data_a, default_a

def use_sm(x_a, y_a):

    b = ones((10000,1))
    xareal_a = hstack((x_a,b))
    print(y_a.dtype)

    print(xareal_a)
    logit_model = sm.Logit(y_a, xareal_a)  
    result = logit_model.fit()
    print(result.summary())
    predicted = result.predict(xareal_a)
    return predicted, xareal_a


def boot_fn(x_a, y_a):
    all_dataset = hstack((x_a,y_a))
    n = all_dataset.shape[0]
    index = arange(n)
    #print(index.shape)
    index_and_const = empty((n,2))
    index_and_const[:,0] = index
    index_and_const[:,1] = 1
    #print(index_and_const, index_and_const.shape)

    data_and_index = hstack((all_dataset, index_and_const))
    #print(data_and_index, data_and_index.shape)
    y_default = y_a[:,0]
    clf = GaussianNB()
    clf.fit(data_and_index, y_default)
    probs = clf.predict_proba(data_and_index)
    print(probs)
    #predicted, xareal_a = use_sm(x_a, y_a)
    #boot_fn(x_a)
    return data_and_index, y_default, probs

def main():
    x_a, y_a = data_loader("Default.csv")
    #use_sm(x_a, y_a)
    dist = norm(loc=2, scale=4)  

    data = dist.rvs(size=10, random_state=rng)
    std_true = dist.std()     

    print(std_true)
    std_sample = np.std(data)  
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    print(std_sample)
    raise SystemExit
    data_and_index, y_default, probs = boot_fn(x_a, y_a)
    #logit_model = sm.Logit(probs, y_default)
    #print(probs.shape, data_and_index.shape)
    #result = logit_model.fit()
    #print(result.summary())
    glmmodel = sm.GLM(probs, data_and_index)
    result = glmmodel.fit
    print(result.summary())

if __name__ == '__main__':
    main()
