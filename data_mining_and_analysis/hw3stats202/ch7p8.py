from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoCV
from scipy.stats import ttest_ind
from scipy.stats import t
from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep, BSpline
import statsmodels.api as sm

#THIS IS CH7 P8 FOR QUESTION 5

'''

To whoever is grading this problem: This file was a huge hot mess and still is a bit of a hot mess I'm sorry. I tried to clean it up a bit.
'''

def data_loader(fname):
    data_a = loadtxt(fname,skiprows=1, usecols=(1,2,3,4,5,6,7), delimiter=',')
    name_v = loadtxt(fname,skiprows = 1, usecols=(0), delimiter=',')
    return data_a, name_v


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

def fit_polynomial_regression(data_a, name_v, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(data_a.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, name_v)
    return model

def fit_polynomial_regression(data_a, name_v, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(data_a.reshape(-1, 1))
def fit_polynomial_regression2(data_a, name_v, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(data_a.reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, name_v)
    return model

def lass_reg(xtrain, ytrain, xtest,ytest):

    alphas = logspace(-2, 2, 5)
    lass_model = LassoCV(alphas=alphas)
    lass_model.fit(xtrain, ytrain)
    ypred = lass_model.predict(xtest)    
    rsq = 1 - (var(ytest-ypred))/(var(ytest))
    test_error = mean((ytest-ypred)**2)
    non_zero= sum(lass_model.coef_ != 0)
    return rsq, test_error, non_zero


def fit_linear_regression(data_a, name_v):
    X_linear = column_stack((ones(len(data_a)), data_a))
    model = LinearRegression()
    model.fit(X_linear, name_v)
    return model

def fit_quadratic_regression(data_a, name_v):
    X_quad = column_stack((ones(len(data_a)), data_a, data_a**2))
    model = LinearRegression()
    model.fit(X_quad, name_v)
    return model

def perform_t_test(linear_model, quadratic_model, data_a, name_v):
    linear_residuals = name_v - linear_model.predict(column_stack((ones(len(data_a)), data_a)))
    quadratic_residuals = name_v - quadratic_model.predict(column_stack((ones(len(data_a)), data_a, data_a**2)))
    t_stat = (linear_residuals.T @ linear_residuals - quadratic_residuals.T @ quadratic_residuals) / len(name_v)
    return t_stat

def compute_cross_val_error(model, X, name_v):
    cv_error = mean(cross_val_score(model, X, name_v, scoring='neg_mean_squared_error', cv=5))
    return -cv_error

def plot_results(data_a, name_v, X_pred, linear_estimate, quadratic_estimate):
    plt.scatter(data_a, name_v, label='Data')
    plt.plot(X_pred, linear_estimate, label='Linear Regression', color='r')
    plt.plot(X_pred, quadratic_estimate, label='Quadratic Polynomial Regression', color='g')
    plt.xlabel('Predictor X')
    plt.ylabel('Response y')
    plt.legend()
    plt.show()

def main():
    data_a, name_v = data_loader('Auto.csv')

    name_v = name_v.astype(float)


    linear_model = fit_linear_regression(data_a, name_v)

    quadratic_model = fit_quadratic_regression(data_a, name_v)

    t_test_stat = perform_t_test(linear_model, quadratic_model, data_a, name_v)

    X = column_stack((data_a, data_a**2))  # Combine linear and quadratic features
    linear_cv_error = compute_cross_val_error(linear_model, X, name_v)
    quadratic_cv_error = compute_cross_val_error(quadratic_model, X, name_v)

    X_pred = linspace(data_a.min(), data_a.max(), 100)
    X_pred_reshaped = column_stack((ones(100), X_pred))  # Add a column of ones for linear regression
    linear_estimate = linear_model.predict(X_pred_reshaped)
    quadratic_estimate = quadratic_model.predict(column_stack((ones(100), X_pred, X_pred**2)))
    plot_results(data_a, name_v, X_pred, linear_estimate, quadratic_estimate)

    print("T-test statistic:", t_test_stat)
    print("Linear CV error:", linear_cv_error)
    print("Quadratic CV error:", quadratic_cv_error)
    polynomial_models = []
    for degree in range(1, 6):
        polynomial_model = fit_polynomial_regression(data_a, name_v, degree)
        polynomial_models.append(polynomial_model)

    linear_residuals = name_v - linear_model.predict(column_stack((ones(len(data_a)), data_a)))
    t_test_stat, p_value = ttest_ind(linear_residuals, zeros(len(linear_residuals)))  # Null hypothesis: linear model has no effect

    polynomial_t_stats = []
    polynomial_p_values = []
    for polynomial_model in polynomial_models:
        polynomial_residuals = name_v - polynomial_model.predict(PolynomialFeatures(degree=polynomial_model.degree).fit_transform(data_a.reshape(-1, 1)))
        t_stat, p_value = ttest_ind(polynomial_residuals, zeros(len(polynomial_residuals)))  # Null hypothesis: polynomial model has no effect
        polynomial_t_stats.append(t_stat)
        polynomial_p_values.append(p_value)

    #x_a, y_v, data_a, name_l  = data_loader('Auto.csv')
    #cor_a = corrcoef(data_a, rowvar=False)
    #print(cor_a.shape)
    #with np.printoptions(precision=4):
    #    print(cor_a)

    #print(x_a,y_v)
    #scatter_matrix(data_a, name_l)

    #print(x_a,y_v)
    #b_v = lin_regression(x_a,y_v, name='Main Regression')
    #yfit_v = fitted_func(x_a, b_v)
    #i_v = abs(b_v).argsort()[::-1]
    #print(f'Coefficients {b_v=}')
    

if __name__ == "__main__":
    main()

