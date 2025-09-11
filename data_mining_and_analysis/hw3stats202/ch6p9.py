from wsgiref.headers import tspecials
from numpy import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoCV


# THIS IS CH6 P9 FOR QUESTION 3
def data_loader(fname):
    data_a = loadtxt(fname,skiprows=1, usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), delimiter=',')
    
    return data_a

def lin_model(x_a, y_v):
    #y_v = X@B
    b_v = linalg.pinv (x_a)@ y_v
    return b_v

def lin_fit(x_a, b_v):
    ypred_v = x_a@b_v
    return ypred_v

def rid_reg(xtrain, ytrain, xtest,ytest):
    alphas = logspace(-2, 2, 5)  


    rid_model = RidgeCV(alphas=alphas, store_cv_values=True)

    rid_model.fit(xtrain, ytrain)
    ypred = rid_model.predict(xtest)
    rsq = 1 - (var(ytest-ypred))/(var(ytest))
    test_error = mean((ytest-ypred)**2)

    return rsq, test_error
def lass_reg(xtrain, ytrain, xtest,ytest):

    alphas = logspace(-2, 2, 5)
    lass_model = LassoCV(alphas=alphas)
    lass_model.fit(xtrain, ytrain)
    ypred = lass_model.predict(xtest)    
    rsq = 1 - (var(ytest-ypred))/(var(ytest))
    test_error = mean((ytest-ypred)**2)
    non_zero= sum(lass_model.coef_ != 0)
    return rsq, test_error, non_zero


def main():
    data_a = data_loader("College.csv")
    #print(data_a)
    #print(len(data_a))
    '''Getting data set up'''
    n = int(0.75*len(data_a))

    xtrain = data_a[:n, 1:]
    xtrainreal = hstack((ones((xtrain.shape[0], 1)), xtrain))
    ytrain = data_a[:n ,:1]

    xtest = data_a[n:, 1:]
    xtestreal = hstack((ones((xtest.shape[0], 1)), xtest))  
    ytest = data_a[n:, :1]
    
    print(xtrainreal.shape)
    print(ytrain.shape)
    print(xtestreal.shape)
    print(ytest.shape)

    '''Doing fit'''
    coef = lin_model(xtrainreal, ytrain)

    print(coef)

    pred_apps = lin_fit(xtestreal, coef)
    print(pred_apps)
    test_error = mean((ytest-pred_apps)**2)

    rsq = 1 - (var(ytest-pred_apps))/(var(ytest))
    #print(rsq)
    print(f'The mse for lin reg is {test_error} and the r squared value for lin is {rsq}')

    rsqrid, mserid = rid_reg(xtrainreal, ytrain, xtestreal, ytest)
    print(f'The r squared value for ridge is {rsqrid} and the mse for ridge regression is {mserid}')

    rsqlass, mselass, nonzero = lass_reg(xtrainreal, ytrain, xtestreal, ytest)
    print(f'The r squared value for lasso is {rsqlass} and the mse for lasso  is {mselass} and {nonzero} coefficents')

if __name__ == '__main__':
    main()