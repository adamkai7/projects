from wsgiref.headers import tspecials
from numpy import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

#THIS IS CH9 P 8 FOR QUESTION 7
def data_loader(fname):
    # 1WeekofPurchase 2StoreID	3PriceCH	4PriceMM	5DiscCH	6DiscMM	7SpecialCH	8SpecialMM	9LoyalCH	10SalePriceMM	11SalePriceCH	12PriceDiff
    #14PctDiscMM	15PctDiscCH	16ListPriceDiff	17STORE
    data_a = loadtxt(fname,skiprows=1, usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17), delimiter=',')
    #0 Purchase 13Store7
    #For store 7 i replaced no with 0 and yes with 1 
    purchase_v = loadtxt(fname, skiprows=1, usecols=(0), delimiter=',', dtype=str)

    return data_a, purchase_v

def sup_vec_class(xtrain, ytrain):
    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
    #https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
    sup_vec_classifier = SVC(C=0.01)
    sup_vec_classifier.fit(xtrain, ytrain)
    support = len(sup_vec_classifier.support_vectors_)
    return sup_vec_classifier, support

def sup_vec_classv2(xtrain, ytrain,bestc):
    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
    #https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
    sup_vec_classifier = SVC(C=bestc)
    sup_vec_classifier.fit(xtrain, ytrain)
    #support = len(sup_vec_classifier.support_vectors_)
    return sup_vec_classifier

def sup_vec_class_rbf(xtrain, ytrain):
    sup_vec_class_radial = SVC(C=0.01, kernel='rbf')
    sup_vec_class_radial.fit(xtrain, ytrain)
    support = len(sup_vec_class_radial.support_vectors_)
    return sup_vec_class_radial, support
def sup_vec_class_rbfv2(xtrain, ytrain,bestcrad):
    #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    #https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
    #https://pythonprogramming.net/linear-svc-example-scikit-learn-svm-python/
    sup_vec_class_best_c_radial = SVC(C=bestcrad, kernel='rbf')    
    sup_vec_class_best_c_radial.fit(xtrain, ytrain)
    #support = len(sup_vec_classifier.support_vectors_)
    return sup_vec_class_best_c_radial

def sup_vec_class_poly(xtrain,ytrain):
    sup_vec_class_poly = SVC(C=0.01, kernel='poly', degree=2)
    sup_vec_class_poly.fit(xtrain,ytrain)
    support = len(sup_vec_class_poly.support_vectors_)
    return sup_vec_class_poly, support 

def sup_vec_class_polyv2(xtrain, ytrain,bestpoly):
    sup_vec_class_best_c_poly = SVC(C=bestpoly, kernel='rbf')    
    sup_vec_class_best_c_poly.fit(xtrain, ytrain)
    #support = len(sup_vec_classifier.support_vectors_)
    return sup_vec_class_best_c_poly

def acc_score(svmclass, xtrain,ytrain,xtest,ytest):
    ytrainpred = svmclass.predict(xtrain)
    trainscoretrain = 1 - accuracy_score(ytrain, ytrainpred)

    ytestpred = svmclass.predict(xtest)
    testscoretest = 1 - accuracy_score(ytest, ytestpred)
    return trainscoretrain, testscoretest

def find_best_c(svmclass,xtrain,ytrain):
    # this part was heavily influenced by 
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    # https://scikit-learn.org/stable/modules/grid_search.html
    #https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/
    #https://stats.stackexchange.com/questions/305201/optimal-grid-search-for-c-in-svm
    #https://www.baeldung.com/cs/ml-svm-c-parameter
    #finding_c = logspace(-2, 1, 5) #5 values from 0.01 to 10
    param_grid = {'C': [ 0.1, 10]}
    grid_search = GridSearchCV(svmclass, param_grid, cv=100)
    grid_search.fit(xtrain, ytrain)
    best_c = grid_search.best_params_['C']
    return best_c
def find_best_c_radial(svmclassradial, xtrain, ytrain):
    #same as above but now with rbf
    param_grid_radial = {'C': [0.01, 10]}
    svm_cv_radial = SVC(kernel='rbf')
    grid_search_radial = GridSearchCV(svm_cv_radial, param_grid_radial, cv=100)
    grid_search_radial.fit(xtrain, ytrain)
    best_c_radial = grid_search_radial.best_params_['C']
    return best_c_radial

def find_best_c_poly(svmclasspoly, xtrain,ytrain):
    param_grid_poly = {'C': [0.01, 10]}
    svm_cv_poly = SVC(kernel='poly', degree=2)
    grid_search_poly = GridSearchCV(svm_cv_poly, param_grid_poly, cv=100)
    grid_search_poly.fit(xtrain, ytrain)
    best_c_poly = grid_search_poly.best_params_['C']
    return best_c_poly

def main():
    data_a, purchase_v  = data_loader('OJ.csv')

   
    xtrain, xtest, ytrain, ytest = train_test_split(data_a, purchase_v, train_size=800, random_state=42)    
    #print(f'xtrain{xtrain} ytrain{ytrain} xtest{xtest}ytest{ytest}')

    '''part b'''
    svmclass, support = sup_vec_class(xtrain,ytrain)

    print(f'Fitted a support vector classifier to the training data using C = 0.01, with Purchase as the response and the other variablesas predictors. There were {support} support points.')

    '''part c'''
    trainscore, testscore = acc_score(svmclass, xtrain,ytrain,xtest,ytest)
    print(f'Training accuracy score of {trainscore} and test accuracy score of {testscore}')
    '''part d'''
    c = find_best_c(svmclass,xtrain,ytrain)
    print(c)
    '''part e'''
    svmclassv2 = sup_vec_classv2(xtrain,ytrain,c)
    newtrainscore, newtestscore = acc_score(svmclassv2, xtrain,ytrain,xtest,ytest)
    print(f'Fitted a support vector classifier to the training data using the best C = {c}, and got Training accuracy score of {newtrainscore} and test accuracy score of {newtestscore}')


    '''now doing with radial'''
    svmclassradial, supportradial = sup_vec_class_rbf(xtrain, ytrain)
    print(f'Fitted a support vector classifier to the training data using C = 0.01, with Purchase as the response and the other variablesas predictors. Used radial. There were {supportradial} support points.')
    trainscoreradial, testscoreradial = acc_score(svmclassradial, xtrain, ytrain, xtest, ytest)
    print(f'Training accuracy score using radial is {trainscoreradial} and test accuracy score of {testscoreradial}')
    bestcrad = find_best_c_radial(svmclassradial,xtrain,ytrain)
    svmclassradialv2 = sup_vec_class_rbfv2(xtrain, ytrain, bestcrad)
    newtrainscorerad, newtestscorerad = acc_score(svmclassradialv2,xtrain,ytrain,xtest,ytest)
    print(f'Fitted a support vector classifier to the training data using the best C = {bestcrad} with radial, and got Training accuracy score of {newtrainscorerad} and test accuracy score of {newtestscorerad}')

    '''now doing with poly'''
    svmclasspoly, supportpoly = sup_vec_class_poly(xtrain, ytrain)
    print(f'Fitted a support vector classifier to the training data using C = 0.01, with Purchase as the response and the other variablesas predictors. Used poly. There were {supportpoly} support points.')
    trainscorepoly, testscorepoly = acc_score(svmclasspoly, xtrain, ytrain, xtest, ytest)
    print(f'Training accuracy score using radial is {trainscorepoly} and test accuracy score of {testscorepoly}')
    bestcpoly = find_best_c_poly(svmclasspoly,xtrain,ytrain)
    svmclasspolyv2 = sup_vec_class_polyv2(xtrain,ytrain,bestcpoly)
    newtrainscorepoly, newtestscorepoly = acc_score(svmclasspolyv2,xtrain,ytrain,xtest,ytest)
    print(f'Fitted a support vector classifier to the training data using the best C = {bestcpoly} with poly, and got Training accuracy score of {newtrainscorepoly} and test accuracy score of {newtestscorepoly}')

if __name__ == '__main__':
    main()



