from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline
from scipy.cluster.hierarchy import complete, fcluster
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

#Chapter 12 question 9  





def data_loader(fname):
    x_a = loadtxt(fname,skiprows=1, usecols=(1,2,3,4), delimiter=',')


    state_v = loadtxt(fname, skiprows=1, usecols=(0), delimiter=',', dtype=str)



    print(state_v)
    #print(data_a[0:,1].shape)
    # We want it in the form  of Y = XB
    # Where Y is the response variable
    # Where X is an array with size nx2 where n is the predictor variable and the other column is a 1
    # B is the coeeficents (slope and intercept) that we are trying to solve for    year_v = data_a[:,1]
  
    
    
    return x_a, state_v

def dendo_construct(x_a, state_v):
    n = state_v.shape[0]

    def label_func(id):
        if id < n:
            return state_v[id]
        return f'{id}'

    d_a = pdist(x_a) 
    print(d_a.shape)
    print(d_a)
    z_a = complete(d_a)
    print(z_a.shape)
    print(z_a)
    plt.figure()
    dn = hierarchy.dendrogram(z_a, leaf_label_func=label_func)
    plt.show()

def z_score(x_a):
    m_v = x_a.mean(axis=0)
    std_v = x_a.std(axis=0)
    z_a = (x_a - m_v)/std_v
    return z_a

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
    fig.savefig('ch12_9.pdf')       
    plt.show()

def main ():
    x_a, state_v  = data_loader('USArrests.csv')
    dendo_construct(x_a, state_v)
    dendo_construct(z_score(x_a), state_v)
    raise SystemExit
    raise SystemExit

    #print(yhat_v)
if __name__ == '__main__':
	main()	


# zeros, full, empty, array, arange, indexing, transpose  

#URLs used: https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram