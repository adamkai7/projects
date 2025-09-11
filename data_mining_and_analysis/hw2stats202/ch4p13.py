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

#Adam Kainikara
#This code is for
#CHAPTER 4 QUESTION 13 
#THIS IS FOR PROBLEM 5
#OF HOMEWORK 2 FOR STANFORD SUMMER SESSION STATS 202


def data_loader(fname):
    data_a = loadtxt(fname,skiprows=1, usecols=(1,2,3,4,5,6,7), delimiter=',')
    direction_v = loadtxt(fname, skiprows=1, usecols=(8), delimiter=',', dtype=str)

    return data_a, direction_v

def summaries0(x_a):
    print("Numerical summaries")
    for i in range(x_a.shape[1]):
        column = x_a[:, i]
        print(f'Column {i + 1}:')
        print(f'   - Min: {min(column)}')
        print(f'   - Max: {max(column)}')
        print(f'   - Mean: {mean(column)}')
        print(f'   - Median: {median(column)}')
        print(f'   - Stand Dev: {std(column)}')
    print(x_a.shape[1])

    summary = empty((8,5))
    storage_l = []

    for i in range(x_a.shape[1]):
        column = x_a[:, i]
        summary = [min(column), max(column), mean(column), median(column), std(column)]
        print(summary)
        storage_l.append(summary)
    return np.array(storage_l)

def calc_summary(x_a) -> dict[str,ndarray]:
    data_d = {
        "min": x_a.min(axis = 0),
        "max": x_a.max(axis = 0),
        "mean": x_a.mean(axis = 0),
        "stand dev": x_a.std(axis = 0),
        "median": median(x_a, axis = 0)
    }
    return data_d

def print_summary(data_d: dict[str,ndarray]):
    for k,v in data_d.items():
        print(f'{k}:', v)

def find_p_values(x_a, direction_v):
    #y_v = direction_v
    #log_reg = sm.Logit(y_v,ee x_a).fieet()

    # Extract the p-values
    #print(log_reg.summary())
    pass
      
def plot_summary(data_d):
    data_d = {'min': data_d['min'], 'max': data_d['max'], 'mean': data_d['mean'], 'stan dev': data_d['stand dev'], 'median': data_d['median']
    }
    nvar = len(data_d["min"])
    print(data_d)
    w = 0.25
    stride = w + 0.05
    initial_change = 0
    labels = ("Min", "Max", "Mean", "Stan Dev", "Median")
    x_v = arange(nvar) * stride + 0
    print(x_v)
    
    fig, ax = plt.subplots()

    for atrribute, measurement in data_d.items():
        #shift = w * initial_change
        rectangle = ax.bar(x_v, measurement, width=w, label = atrribute)    
        x_v += nvar * stride
        ax.bar_label(rectangle, fmt = '%.02f', rotation = 45, padding=2)
        #i  nitial_change += 0.25
        #ax.set_xticks(x_v , labels)


    plt.show()

def summaries(x_a):
    print("Numerical summaries")
    #min_x_a = x_a.min(axis = 0)
    print(f' Min:{x_a.min(axis = 0)}, Max{x_a.max(axis = 0)}, Mean{x_a.mean(axis = 0)}, Stand Dev {x_a.std(axis = 0)}, Median{median(x_a, axis = 0)}')

def logi_reg(x_a, direction_v):
    predictors_a = x_a[:,1:7]
    #print(predictors_a)
    #print(x_a.shape)
    response_v = direction_v

    #response_v = x_a[:,8]
    #print(response_v)
    logreg = LogisticRegression()  
    logreg.fit(predictors_a, response_v)
    coefficients = logreg.coef_

    print(coefficients)

def make_prediction(x_a, direction_v):
    clf = LogisticRegression()
    clf.fit(x_a, direction_v)
    ypredict_v = clf.predict(x_a)
    return ypredict_v


def compute_confusion_mat(ypredict_v, direction_v):
    truey_v = direction_v
    confu_mat = confusion_matrix(truey_v, ypredict_v)
    print(confu_mat)
    return confu_mat

def lda_prediction(x_v, y_v):
    #This only served to help me write the code, I did it in main otherwise
    clf = lda()
    clf.fit(x_v, y_v)
    ypredict_ldatrain_v = clf.predict(x_v)
    ypredict_ldatest_v = clf.predict(x_v)
    #Rembr to do the confusion matrix after
    return ypredict_ldatrain_v, ypredict_ldatest_v
    #What I ended up doing
    clf = lda()
    clf.fit(xtrain_v, ytrain_v)

    ypredict_ldatrain_v = clf.predict(xtrain_v)
    print("Confusion Matrix of the training data with Lag2 as the only predictor but instead used LDA")
    compute_confusion_mat(ypredict_ldatrain_v, ytrain_v)

    ypredict_ldatest_v = clf.predict(xtest_v)
    print("Confusion Matrix of the test data with Lag2 as the only predictor but instead used LDA")
    compute_confusion_mat(ypredict_ldatest_v, ytest_v)

def qda_prediction(x_v, y_v):
    #This only served to help me write the code, I did it in main otherwise
    clf = qda()
    clf.fit(x_v, y_v)
    ypredict_ldatrain_v = clf.predict(x_v)
    ypredict_ldatest_v = clf.predict(x_v)
    #Rembr to do the confusion matrix after
    return ypredict_ldatrain_v, ypredict_ldatest_v

    #What I ended up doing
    clf = qda()
    clf.fit(xtrain_v, ytrain_v)
    ypredict_qdatrain_v = clf.predict(xtrain_v)
    print("Confusion Matrix of the training data with Lag2 as the only predictor but instead used QDA")
    compute_confusion_mat(ypredict_qdatrain_v, ytrain_v)

    ypredict_qdatest_v = clf.predict(xtest_v)
    print("Confusion Matrix of the test data with Lag2 as the only predictor but instead used QDA")
    compute_confusion_mat(ypredict_qdatest_v, ytest_v)
def naiv_prediction(x_v, y_v):
    #This onl served to elp me write the code, I did it in main otherwise
    clf = GaussianNB()
    clf.fit(x_v, y_v)
    ypredict_nbtrain_v = clf.predict(x_v)
    ypredict_nbtrest_v = clf.predict(x_v)
    return ypredict_nbtrain_v, ypredict_nbtrest_v
def knn_prediction(x_v, y_v):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_v, y_v)
    ypredict_knntrain_v = neigh.predict(x_v)
    ypredict_knntest_v = neigh.predict(x_v)
    return ypredict_knntrain_v, ypredict_knntest_v
   
def main():
    x_a, y_v = data_loader("Weekly.csv")
 
    xtrain_v = x_a[:,1:2][:985]
    xtest_v = x_a[:,1:2][985:]

    ytrain_v = y_v[:985]
    ytest_v = y_v[985:]


    '''
    Above is the test and training data for x and y for the remainder of this problem. We will first train then do the test data.
    Then do the confusion matrix
    '''

    '''
    This first part (below) is fitting the training data and then predicitng the y value based on the training data. I
    t then computes the confusion matrix based on the training data
    '''
   

    clf = LogisticRegression()
    clf.fit(xtrain_v, ytrain_v)
   

    ytrain_pred_v = clf.predict(xtrain_v)
    print("Confusion Matrix of the training data with Lag2 as the only predictor")
    compute_confusion_mat(ytrain_pred_v, ytrain_v)

  
    '''
    This second part (below) is getting the predicited y value and computing 
    the confusion matrix based on the fit found earlier and the test data.
    '''

    ytest_pred_v = clf.predict(xtest_v)
    print("Confusion Matrix of the test data with Lag2 as the only predictor")
    compute_confusion_mat(ytest_pred_v, ytest_v)

    """
    ============================================================================================================================================
    NOW DOING QDA
    """
    clf = lda()
    clf.fit(xtrain_v, ytrain_v)
    ypredict_ldatrain_v = clf.predict(xtrain_v)
    print("Confusion Matrix of the training data with Lag2 as the only predictor but instead used LDA")
    compute_confusion_mat(ypredict_ldatrain_v, ytrain_v)

    ypredict_ldatest_v = clf.predict(xtest_v)
    print("Confusion Matrix of the test data with Lag2 as the only predictor but instead used LDA")
    compute_confusion_mat(ypredict_ldatest_v, ytest_v)

    """
    ============================================================================================================================================
    NOW DOING QDA
    """
    clf = qda()
    clf.fit(xtrain_v, ytrain_v)
    ypredict_qdatrain_v = clf.predict(xtrain_v)
    print("Confusion Matrix of the training data with Lag2 as the only predictor but instead used QDA")
    compute_confusion_mat(ypredict_qdatrain_v, ytrain_v)

    ypredict_qdatest_v = clf.predict(xtest_v)
    print("Confusion Matrix of the test data with Lag2 as the only predictor but instead used QDA")
    compute_confusion_mat(ypredict_qdatest_v, ytest_v)
    """
    ============================================================================================================================================
    NOW DOING NAIVE BAEES
    """
    clf = GaussianNB()
    clf.fit(xtrain_v, ytrain_v)
    ypredict_nbtrain_v = clf.predict(xtrain_v)
    print("Confusion Matrix of the training data with Lag2 as the only predictor but instead used NAIVE BAYES")
    compute_confusion_mat(ypredict_nbtrain_v, ytrain_v)

    ypredict_nbtest_v = clf.predict(xtest_v)
    print("Confusion Matrix of the test data with Lag2 as the only predictor but instead used NAIVE BAYES")
    compute_confusion_mat(ypredict_nbtest_v, ytest_v)

    """
    ============================================================================================================================================
    NOW DOING KNN
    """
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(xtrain_v, ytrain_v)
    ypredict_knntrain_v = neigh.predict(xtrain_v)
    print("Confusion Matrix of the training data with Lag2 as the only predictor but instead used KNN")
    compute_confusion_mat(ypredict_knntrain_v, ytrain_v)

    ypredict_knntest_v = neigh.predict(xtest_v)
    print("Confusion Matrix of the test data with Lag2 as the only predictor but instead used KNN")
    compute_confusion_mat(ypredict_knntest_v, ytest_v)

    """
    ============================================================================================================================================
    NOW DOING LDA WITH A TWIST
    """
    xtrain2_v = x_a[:,1:3][:985]
    xtest2_v = x_a[:,1:3][985:]
    ytrain_v = y_v[:985]
    ytest_v = y_v[985:]
    clf = lda()
    clf.fit(xtrain2_v, ytrain_v)
    ypredict_ldatrain2_v = clf.predict(xtrain2_v)
    print("Confusion Matrix of the training data with Lag2 and Lag3 as the only predictors but instead used LDA")
    compute_confusion_mat(ypredict_ldatrain2_v, ytrain_v)

    ypredict_ldatest2_v = clf.predict(xtest2_v)
    print("Confusion Matrix of the test data with Lag2 and Lag3 as the only predictors but instead used LDA")
    compute_confusion_mat(ypredict_ldatest2_v, ytest_v)
    """
    ============================================================================================================================================
    NOW DOING QDA WITH A TWIST
    """
    clf = qda()
    clf.fit(xtrain2_v, ytrain_v)
    ypredict_qdatrain2_v = clf.predict(xtrain2_v)
    print("Confusion Matrix of the training data with Lag2 and Lag 3 as the only predictors but instead used QDA")
    compute_confusion_mat(ypredict_qdatrain2_v, ytrain_v)

    ypredict_qdatest2_v = clf.predict(xtest2_v)
    print("Confusion Matrix of the test data with Lag2 and Lag3 as the only predictors but instead used QDA")
    compute_confusion_mat(ypredict_qdatest2_v, ytest_v)
    """
    ============================================================================================================================================
    NOW DOING NAIVE BAEES WITH A TWIST
    """
    clf = GaussianNB()
    clf.fit(xtrain2_v, ytrain_v)
    ypredict_nbtrain2_v = clf.predict(xtrain2_v)
    print("Confusion Matrix of the training data with Lag2 and Lag 3 as the only predictor but instead used NAIVE BAYES")
    compute_confusion_mat(ypredict_nbtrain2_v, ytrain_v)

    ypredict_nbtest2_v = clf.predict(xtest2_v)
    print("Confusion Matrix of the test data with Lag2 and Lag 3 as the only predictor but instead used NAIVE BAYES")
    compute_confusion_mat(ypredict_nbtest2_v, ytest_v)
    

    """
    ============================================================================================================================================
    NOW DOING KNN WITH A TWIST
    """
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(xtrain2_v, ytrain_v)
    ypredict_knntrain2_v = neigh.predict(xtrain2_v)
    print("Confusion Matrix of the training data with Lag2 and Lag 3 as the only predictor but instead used KNN")
    compute_confusion_mat(ypredict_knntrain2_v, ytrain_v)

    ypredict_knntest2_v = neigh.predict(xtest2_v)
    print("Confusion Matrix of the test data with Lag2 and Lag 3 as the only predictor but instead used KNN")
    compute_confusion_mat(ypredict_knntest2_v, ytest_v)

    clf = LogisticRegression()
    clf.fit(xtrain2_v, ytrain_v)
   

    ytrain_pred2_v = clf.predict(xtrain2_v)
    print("Confusion Matrix of the training data with Lag2 and Lag3 as the only predictors")
    compute_confusion_mat(ytrain_pred2_v, ytrain_v)


    ytest_pred2_v = clf.predict(xtest2_v)
    print("Confusion Matrix of the test data with Lag2 and Lag 3 as the only predictor")
    compute_confusion_mat(ytest_pred2_v, ytest_v)




    #print(xtest_a)
    #print(ytrain_v)

    #ytest_v = y_v[:ntest]
    #xtrain_v = x_a[ntest:]

    #confusion_pract()
    #print(x_a)
    #logi_reg(x_a, y_v)
    #find_p_values(x_a, direction_v)
    #summaries(x_a)
    summary = summaries(x_a)
    #calc_summary(x_a)
    data_dict = calc_summary(x_a)
    print_summary(data_dict)
    plot_summary(data_dict)


    clf = LogisticRegression()
    clf.fit(xtrain_v, ytrain_v)
    ytrain_pred_v = clf.predict(xtrain_v)
    compute_confusion_mat(ytrain_pred_v, ytrain_v)
    ytest_pred_v = clf.predict(xtest_v)
    compute_confusion_mat(ytest_pred_v, ytest_v)
    print('Done-----')

    make_prediction(x_a, y_v)
    ypredict_v = make_prediction(x_a, y_v)
    print(ypredict_v)
    compute_confusion_mat(ypredict_v, y_v)
    #print(type(summary))
    #print(summary.shape)
    #plot_summaries(summary)



#Some Links I used
#https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html#sphx-glr-auto-examples-classification-plot-lda-qda-py
#https://stackoverflow.com/questions/46775155/importerror-no-module-named-sklearn-lda

if __name__ == '__main__':
	main()	

