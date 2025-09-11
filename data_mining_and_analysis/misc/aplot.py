# this is just saving usefull stuff, mostly plots 

# do this in the v env cause it wont work

from turtle import width
from urllib import response
from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep, BSpline
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
