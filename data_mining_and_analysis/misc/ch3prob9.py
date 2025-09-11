from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep, BSpline
import statsmodels.api as sm


#Adam Kainikara
#This code is for
#CHAPTER 3 QUESTION 9 

def data_loader(fname):
    data_a = loadtxt(fname,skiprows=1, usecols=(0,1,2,3,4,5,6,7), delimiter=',')

    #mpg	cylinders	displacement	horsepower	weight	acceleration	year	origin	name


    #print(data_a.shape)
    #print(data_a[0:,1].shape)
    # We want it in the form  of Y = XB
    # Where Y is the response variable
    # Where X is an array with size nx2 where n is the predictor variable and the other column is a 1
    # B is the coeeficents (slope and intercept) that we are trying to solve for    year_v = data_a[:,1]
  
    n, m = data_a.shape
    pred_a = data_a[:, 1:]

    x_a = empty([n,pred_a.shape[1]+1], dtype=float64)
    x_a[:,:-1] =  pred_a
    x_a[:,-1] = 1
    name_l = ['mpg',	'cylinders',	'displacement',	'horsepower',	'weight',	'acceleration',	'year',	'origin']

    #print(x_v.shape)

    y_v = data_a[:,0]
    
    return x_a, y_v, data_a, name_l

def scatter_matrix(data_a, name_l):
    n, p = data_a.shape
    fig, axs = plt.subplots(4, 7)
    ax_l = list(axs.flat) 
    mpl.rcParams['figure.autolayout'] = True
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

    #mpl.rc('font', **font)
    for i in range(p):
        for j in range(i+1,p):
            print(i, j, name_l[i], name_l[j])
            x_v = data_a[:,i]
            y_v = data_a[:,j]
            ax = ax_l.pop(0)
            ax.scatter(x_v, y_v, s=2**2)
            title = f'{name_l[i]} vs {name_l[j]}'
            ax.set_title(title[:25])
    plt.tight_layout()
    plt.show()

def diag_plot(y_v, yfit_v):
    res_v = y_v - yfit_v
    i_v = yfit_v.argsort()
    yfit1_v = yfit_v[i_v]
    res1_v = res_v[i_v]
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.scatter(yfit1_v,res1_v)
    #spline = CubicSpline(yfit1_v, res1_v)
    #res2_v = spline(yfit1_v)
    tck = splrep(yfit1_v, res1_v, s=3500)
    res2_v = BSpline(*tck)(yfit1_v)
    #ax.plot(yfit1_v, res1_v)
    ax.plot(yfit1_v, res2_v)

    plt.show()

def transform1_reg(x_a):
    p_a = x_a[:,2:3]
    p_a = (p_a)**2
    
    return hstack((p_a,x_a))

def transform2_reg(x_a):
    p_a = x_a[:,2:3]
    p_a = (p_a)**0.5
    
    return hstack((p_a,x_a))

def transform3_reg(x_a):
    p_a = x_a[:,2:3]
    p_a = log((p_a))
    
    return hstack((p_a,x_a))

def transform4_reg(x_a):
    p_a = x_a[:,2:4]
    p_a = product(p_a,axis=1,keepdims=True)
    
    return hstack((p_a,x_a))

def transform5_reg(x_a):
    p_a = x_a[:,2:5]
    p_a = product(p_a,axis=1,keepdims=True)
    
    return hstack((p_a,x_a))


def interaction_study(x_a, y_v, name):
    b_v = lin_regression(x_a,y_v,name=name)
    yfit_v = fitted_func(x_a, b_v)
    print(f'{name}: Coefficients {b_v=}')
    print(f'{name}: r squared = ', r_square(y_v,yfit_v))

    


def lin_regression(x_a,y_v,name=''):
    if name:
        print(f'\n\n----------------------- {name} --------------------')
        model = sm.OLS(y_v, x_a)
        results = model.fit()
        print(results.summary())
    #Using stats models to get p value even though I did my own regression 
    # y_v = X@B
    b_v = linalg.pinv(x_a)@y_v
    #print(b)
    return b_v 

def fitted_func(x_a,b_v):
    yfit_v = x_a@b_v
    # with np.printoptions(precision=2):
    #     print(f'predicted mpg of cars {yfit_v=}')
    return yfit_v

def r_square(y_v,yfit_v):
    # This function is to find the r squared value 
    # This will be calcualted by doing 1 - variance of (actual - predicited)/variance of actual
    rsq = 1 - (var(y_v-yfit_v))/(var(y_v))

    return rsq

def main ():
    x_a, y_v, data_a, name_l  = data_loader('Auto.csv')
    cor_a = corrcoef(data_a, rowvar=False)
    print(cor_a.shape)
    with np.printoptions(precision=4):
        print(cor_a)

    #print(x_a,y_v)
    scatter_matrix(data_a, name_l)

    #print(x_a,y_v)
    b_v = lin_regression(x_a,y_v, name='Main Regression')
    yfit_v = fitted_func(x_a, b_v)
    i_v = abs(b_v).argsort()[::-1]
    print(f'Coefficients {b_v=}')
    print('Coefficients:', [(name, b) for name, b in zip(name_l, b_v)])
    print("pred_influence",[name_l[i] for i in i_v])
    print('r squared = ', r_square(y_v,yfit_v))
    diag_plot(y_v, yfit_v)
    
    with np.printoptions(precision=2):
        interaction_study(x_a, y_v, 'control')
        interaction_study(transform1_reg(x_a),y_v, 'Horse Power Squared')
        interaction_study(transform2_reg(x_a),y_v, 'Horse Power Square Root')
        interaction_study(transform3_reg(x_a),y_v, 'Horse Power Log')
        interaction_study(transform4_reg(x_a),y_v, 'Horse Power * Weight')
        interaction_study(transform4_reg(x_a),y_v, 'Horse Power * Weight * Acceleration')

    raise SystemExit

    #print(yhat_v)
if __name__ == '__main__':
	main()	


# zeros, full, empty, array, arange, indexing, transpose  

#URLs used: https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
#