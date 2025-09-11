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


#Adam Kainikara
#This code is for
#CHAPTER 5 QUESTION 5
#THIS IS PROBLEM 7
#OF HOMEWORK 2 FOR STANFORD SUMMER SESSION STATS 202

def data_loader(fname):
    num_data_a = loadtxt(fname,skiprows=1, usecols=(2,3), delimiter=',')
    defa_student_a = loadtxt(fname, skiprows=1, usecols=(0,1), delimiter=',', dtype=str)

    return num_data_a, defa_student_a

def logi_reg(x_a, y_a):
    predictors_a = x_a
    response_v = y_a[:,0]
    #print(predictors_a, response_v)
    logreg = LogisticRegression()  
    logreg.fit(predictors_a, response_v)
    coefficients = logreg.coef_
    print(coefficients)
def new_logi_reg(x_a, y_a):
    pass

def compute_confusion_mat(ypredict_v, direction_v):
    truey_v = direction_v
    confu_mat = confusion_matrix(truey_v, ypredict_v)
    print(confu_mat)
    return confu_mat

def main():

    x_a, y_a = data_loader("Default.csv")
    
    
    ydefault_v = y_a[:,0]
    print(y_a[:,1])
    #ysudent = [1 if x == "yes" else 0 for x in x_a[:,1]]
    ystudent = [1 if x == "Yes" else 0 for x in y_a[:,1]]
    #print(ystudent)

    #print(x_a)
    student_a = array(ystudent)
    #print(student_a.shape)
    xall_a = array([x_a[:,0], x_a[:,1],student_a])
    realx_a = transpose(xall_a)
    print(realx_a.shape)
    print(realx_a)

    xalltrain_a = realx_a[:5000]
    xallvalid_a = realx_a[5000:]
    print(xalltrain_a, xallvalid_a)
    ytrain_v = ydefault_v[:5000]
    yvalid_v = ydefault_v[5000:]
    clf = LogisticRegression()
    clf.fit(xalltrain_a, ytrain_v)
    ytrain_pred_v = clf.predict(xalltrain_a)
    yvalid_pred_v = clf.predict(xallvalid_a)
    coefficients = clf.coef_
    print(coefficients)
    compute_confusion_mat(ytrain_pred_v, yvalid_pred_v)
    raise SystemExit

    xall_a = ([x_a],[ysudent])
    print(xall_a)
    #logi_reg(x_a,y_a)

  
    xtrain_a = x_a[:5000]
    xvalid_a = x_a[5000:]
    ytrain_v = ydefault_v[:5000]
    yvalid_v = ydefault_v[5000:]
    


    clf = LogisticRegression()
    clf.fit(xtrain_a, ytrain_v)
    ytrain_pred_v = clf.predict(xtrain_a)
    yvalid_pred_v = clf.predict(xvalid_a)
    coefficients = clf.coef_
    print(coefficients)
    #compute_confusion_mat(ytrain_pred_v, yvalid_pred_v)


    clf = GaussianNB()
    clf.fit(xtrain_a, ytrain_v)
    posterior_probs = clf.predict_proba(xvalid_a)
    predictions = (posterior_probs > 0.5)
    print(predictions)
    accuracy = accuracy_score(yvalid_v, predictions)
    print("Accuracy:", accuracy)
    #print(xtrain_a, ytrain_v)
    raise SystemExit

    ytrain_v = y_v[:985]
    #ytest_v = y_v[985:]




if __name__ == '__main__':
    main()