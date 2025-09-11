from numpy import *
import matplotlib.pyplot as plt
from matplotlib import patheffects
from statistics import linear_regression
from scipy.interpolate import CubicSpline


#THIS CODE IS FOR THE 5 THINGS GRAPH


fig, ax = plt.subplots(figsize=(15, 15))

#x x^2 constant
def powxy(x, y):
    return [x**2, x, 1, y]
def powxy2(x, y):
    return [x**-3, x**-2, x**-1, 1, y]
def startdata():
    pass
    data_a = array([
        powxy(0,4),
        #powxy(11,90),
        powxy(15,170),
        powxy(20,400)
        ])
   # yvalue_start = array([
    #    powxy[4]
   # ])
    X_a = data_a[:,:-1]
    Y_a = data_a[:,-1]
    print(X_a.shape)
    print(Y_a.shape)
    Xinv_a = linalg.pinv(X_a) 
    coefs = Xinv_a @ Y_a
    print(coefs)

    x_v = linspace(0.5,20,400)
    y_v = coefs[0] * (x_v **2) + coefs[1] * (x_v ** 1) + coefs[2]    

    #y_v = coefs[0] * (x_v **3 + coefs[1] * (x_v ** 2) + coefs[2] * (x_v ) + coefs[3]   
    print(y_v)
    fig[0, 0].plot(x_v, y_v, label='var') 
    ax.legend()

    plt.show()



def startdata_cubic():

    data_a = array([
        (0,4),
        (5,20),
        (15,140),
        (20,400)
        ])
   # yvalue_start = array([
    #    powxy[4]
   # ])
    X_v = data_a[:,0]
    Y_v = data_a[:,1]
    spline = CubicSpline(X_v, Y_v)
    
    
    x1_v = linspace(0.5,20,400)
    y1_v = spline(x1_v)

    #y_v = coefs[0] * (x_v **3 + coefs[1] * (x_v ** 2) + coefs[2] * (x_v ) + coefs[3]   
    print(y1_v)
    ax.plot(x1_v, y1_v, label='variance') 
    ax.legend()


def bias_cubic():

    data_a = array([
        (0,400),
        (5,140),
        (15,20),
        (20,4)
        ])
   # yvalue_start = array([
    #    powxy[4]
   # ])
    X_v = data_a[:,0]
    Y_v = data_a[:,1]
    spline = CubicSpline(X_v, Y_v)
    
    
    x1_v = linspace(0.5,20,400)
    y1_v = spline(x1_v)

    #y_v = coefs[0] * (x_v **3 + coefs[1] * (x_v ** 2) + coefs[2] * (x_v ) + coefs[3]   
    print(y1_v)
    ax.plot(x1_v, y1_v, label='squared bias') 

    ax.legend()

def trainerror_cubic():

    data_a = array([
        (0,340),
        (5,210),
        (10,110),
        (15,60),
        (20,15)
        ])
   # yvalue_start = array([
    #    powxy[4]
   # ])
    X_v = data_a[:,0]
    Y_v = data_a[:,1]
    spline = CubicSpline(X_v, Y_v)
    
    
    x1_v = linspace(0.5,20,400)
    y1_v = spline(x1_v)

    #y_v = coefs[0] * (x_v **3 + coefs[1] * (x_v ** 2) + coefs[2] * (x_v ) + coefs[3]   
    print(y1_v)
    ax.plot(x1_v, y1_v, label='training error') 

    ax.legend()

def testerror_cubic():

    data_a = array([
        (0,350),
        (5,250),
        (10,140),
        (15,210),
        (20,340)
        ])
   # yvalue_start = array([
    #    powxy[4]
   # ])
    X_v = data_a[:,0]
    Y_v = data_a[:,1]
    spline = CubicSpline(X_v, Y_v)
    
    
    x1_v = linspace(0.5,20,400)
    y1_v = spline(x1_v)

    #y_v = coefs[0] * (x_v **3 + coefs[1] * (x_v ** 2) + coefs[2] * (x_v ) + coefs[3]   
    print(y1_v)
    ax.plot(x1_v, y1_v, label='test error') 

    ax.legend()

def irrerror():
    ax.axhline(y=200, xmin = 0.05, xmax = 0.95, label='irreducible error')
    ax.legend()
def bias():
    data_a = array([
        powxy2(2,25),
        powxy2(3,16),
        powxy2(4,9),
        powxy2(5,4)
        ])
   # yvalue_start = array([
    #    powxy[4]
   # ])
    X_a = data_a[:,:-1]
    Y_a = data_a[:,-1]
    print(X_a.shape)
    print(Y_a.shape)
    Xinv_a = linalg.pinv(X_a) 
    coefs = Xinv_a @ Y_a
    print(coefs)

    x_v = linspace(1,20,400)
    y_v = coefs[0] * (x_v **3) + coefs[1] * (x_v ** 2) + coefs[2] * (x_v ) + coefs[3]   
    print(y_v)
    ax.plot(x_v, y_v, label='bias') 
    ax.legend()

    plt.show()
def main():
    plt.xlabel("flexibility")
    plt.ylabel("value")
    startdata_cubic()
    bias_cubic()
    trainerror_cubic()
    testerror_cubic()
    irrerror()
    plt.show()
    #startdata()
    #bias()
if __name__ == '__main__':
	main()	

#y_v = (coefs[0] * x_v) + ((coefs[1]) * 2) + coefs[3]

#https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axhline.html