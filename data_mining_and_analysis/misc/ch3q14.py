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
#CHAPTER 3 QUESTION 14

def data_loader():
    random.seed(1)
    x1_v = random.uniform(size=100)
    x2_v = 0.5*x1_v + random.normal(size=100)/10
    y_v = 2+2*x1_v+0.3*x2_v+random.normal(size=100)
    x_a = empty((100,3), dtype=float64)
    x_a[:,0] = 1
    x_a[:,1] = x1_v
    x_a[:,2] = x2_v
    return x_a, y_v

def scatter_plot(x1_v,x2_v):
    plt.scatter(x1_v,x2_v)
    plt.show()

def rand_var_study(x_a, y_v, name):
    b_v = lin_regression(x_a,y_v,name=name)
    yfit_v = fitted_func(x_a, b_v)
    print(f'{name}: Coefficients {b_v=}')
    print(f'{name}: r squared = ', r_square(y_v,yfit_v))

def transform1_x1_only(x_a):
    #X1 and constant only 
    return hstack((x_a[:,0:1],x_a[:,1:2]))


def transform1_x2_only(x_a):
    #X2 and constant only 
    return hstack((x_a[:,0:1],x_a[:,2:]))

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
    b_v = lin_regression(x_a,y_v, name=name)
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
    x_a, y_v = data_loader()
    b_v = lin_regression(x_a,y_v,name='Main Regression')
    print('beta values', b_v)    
    scatter_plot(x_a[:,1],x_a[:,2])
    cor_a = corrcoef(x_a[:,1:], rowvar=False)
    print(cor_a.shape)
    with np.printoptions(precision=4):
        print(cor_a)
    rand_var_study(x_a,y_v,'Control')
    rand_var_study(transform1_x1_only(x_a),y_v,'X1 Only')
    rand_var_study(transform1_x2_only(x_a),y_v,'X2 Only')

    x1_a = vstack((x_a,array([[1,0.1,0.8]])))
    y1_v = hstack((y_v,array([6])))
    rand_var_study(x_a,y_v,'New Control')
    rand_var_study(transform1_x1_only(x1_a),y1_v,'New X1 Only')
    rand_var_study(transform1_x2_only(x1_a),y1_v,'New X2 Only')

    raise SystemExit

    #print(x_a,y_v)
    #scatter_matrix(data_a, name_l)

    #print(x_a,y_v)
    yfit_v = fitted_func(x_a, b_v)
    i_v = abs(b_v).argsort()[::-1]
    print(f'Coefficients {b_v=}')
    print('Coefficients:', [(name, b) for name, b in zip(name_l, b_v)])
    print("pred_influence",[name_l[i] for i in i_v])
    print('r squared = ', r_square(y_v,yfit_v))
    
    #diag_plot(y_v, yfit_v)
    interaction_study(x_a, y_v, 'control')
    interaction_study(transform1_reg(x_a),y_v, 'Horse Power Squared')
    interaction_study(transform2_reg(x_a),y_v, 'Horse Power Square Root')
    interaction_study(transform3_reg(x_a),y_v, 'Horse Power Log')
    interaction_study(transform4_reg(x_a),y_v, 'Horse Power * Weight')
    interaction_study(transform4_reg(x_a),y_v, 'Horse Power * Weight * Acceleration')


    #print(yhat_v)
if __name__ == '__main__':
	main()	


# zeros, full, empty, array, arange, indexing, transpose  

#URLs used: https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
#