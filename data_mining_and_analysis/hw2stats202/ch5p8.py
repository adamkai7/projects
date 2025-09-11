from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep, BSpline
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut

def data_loader():
    #random.seed(1)

    rng = np.random.default_rng(100)
    x_v = rng.normal(size = 100)
    y_v = (x_v) - (2 * x_v**2) + (rng.normal(size = 100))
    print(x_v.shape, y_v.shape)

    x_a = empty((100,5), dtype=float64)
    x_a[:,0] = 1
    x_a[:,1] = x_v
    x_a[:,2] = (x_v)**2
    x_a[:,3] = (x_v)**3
    x_a[:,4] = (x_v)**4
    #print("x_v", x_v)
    #print("x_a", x_a)
    return x_a, x_v, y_v

def data_scatterplot(x_v,y_v):
    plt.scatter(x_v,y_v)
    plt.show()

def line_lin_fit(x_a, y_v):
    #y_v = X@B
    b_v = linalg.pinv(x_a[:,0:2])@y_v
    print(b_v)

def line_loocv_fit(x_a, y_v):
    loo = LeaveOneOut()
    loo.get_n_splits(x_a)
    degree_v = arange(1,5)
    
    result_l = []

    for degree in degree_v:

        for train_i_v, test_i_v in loo.split(x_a):
            xtrain_a = x_a[train_i_v,:degree+1]
            ytrain_v = y_v[train_i_v]
            b_v = linalg.pinv(xtrain_a)@ytrain_v
            #print(b_v)

            yfit_v = x_a[:,:degree+1] @ b_v
            mse = ((y_v - yfit_v)**2).mean()
            result_l.append((b_v, mse))
            #print(mse)
    return result_l


def quad_lin_fit(x_a, y_v):
    #y_v = X@B
    b_v = linalg.pinv(x_a[:,0:3])@y_v
    print(b_v)
def cubic_lin_fit(x_a, y_v):
    #y_v = X@B
    b_v = linalg.pinv(x_a[:,0:4])@y_v
    print(b_v)
def xtofour_lin_fit(x_a, y_v):
    #y_v = X@B
    b_v = linalg.pinv(x_a[:,0:5])@y_v
    print(b_v)
def pvalue(x_a,y_v,name=''):
    if name:
        print(f'\n\n----------------------- {name} --------------------')
        model = sm.OLS(y_v, x_a)
        results = model.fit()
        print(results.summary())
    #Using stats models to get p value even though I did my own regression 
def main():
    data_loader()
    x_a, x_v,  y_v = data_loader()
    #data_scatterplot(x_a, y_v)
    line_lin_fit(x_a, y_v)
    quad_lin_fit(x_a, y_v)
    cubic_lin_fit(x_a, y_v)
    xtofour_lin_fit(x_a, y_v)
    print("This seperates normal and LOOCV")
    mod_mse = line_loocv_fit(x_a, y_v)
    #smse_l = sorted(mod_mse, key = lambda x_t: x_t[1])
    #That sorted all the models, and found the one and its coefficents that produced the lowest mean squared error value
    #smse_l = sorted(mod_mse, key = lambda x_t: x_t[1]+x_t[0].shape!=2 *100000)
    #This one aimed at finding the linear model with the lowest mean squared error value. This was done by using the kornicer delta. 
    #If the shape of the first term (where we had all the coeffients) was not 2 (!= is not equal ) (intercept and slope) it would increase the MSE
    #By 100,000 which means it wouldnt show up cause it is sorted by decreasing MSE
    
    for degree in range(1,5):
        #now changed it a bit so that it loops and prints what i need for all of them
        smse_l = sorted(mod_mse, key = lambda x_t: x_t[1]+(x_t[0].shape[0]!=degree+1)*100000)

        print('Degree', degree, smse_l[0])
    #print(mod_mse_l[:10])
    #print(mod_mse_l[100:110])


if __name__ == '__main__':
	main()	

