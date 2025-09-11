from numpy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from scipy import stats
from scipy.stats import bootstrap
def data_loader(fname):
    data_a = loadtxt(fname,skiprows=1, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13), delimiter=',')

    return data_a

def pop_mean(data_a):
    #Want population mean of medv
    medv_v = data_a[:,13]
    muhat = medv_v.mean()
    #print(muhat)
    #print(medv_v)
    return medv_v, muhat

def stand_error_muhat(medv_v, muhat):
    #Hint: We can compute the standard error of the sample mean by dividing the sample standard deviation by the square root of the 
    # number of observations.
    stdmuhat = medv_v.std()
    n = medv_v.shape[0]

    stand_err_muhat = stdmuhat/(n**0.5)
    print(stand_err_muhat)
    return stand_err_muhat

def newboostrapstderror(medv_v):
    #I am not sure if i coded a method of bootstrap correctly. I referenced the following websites
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    #https://www.statology.org/bootstrapping-in-python/
    #https://medium.com/swlh/bootstrap-sampling-using-pythons-numpy-85822d868977

    nbootstrap = 1000
    bootstrapmeans = []
    for _ in range(nbootstrap):
        bootstrap_sample = random.choice(medv_v, size=len(medv_v), replace=True)
        bootstrap_sample_mean = mean(bootstrap_sample)
        bootstrapmeans.append(bootstrap_sample_mean)

    standard_error = np.std(bootstrapmeans)

    print("Standard Error of µ̂ using Bootstrap:", standard_error)
    return standard_error
def newboostrapstderror_median(medv_v):
    #I am not sure if i coded a method of bootstrap correctly. I referenced the following websites
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    #https://www.statology.org/bootstrapping-in-python/
    #https://medium.com/swlh/bootstrap-sampling-using-pythons-numpy-85822d868977

    nbootstrap = 1000
    bootstrapmedian = []
    for _ in range(nbootstrap):
        bootstrap_sample = random.choice(medv_v, size=len(medv_v), replace=True)
        bootstrap_sample_median = median(bootstrap_sample)
        bootstrapmedian.append(bootstrap_sample_median)

    standard_error = std(bootstrapmedian)

    print("Standard Error of mu hat median using Bootstrap:", standard_error)
    return standard_error
def newbootstrapstderror_tenpercen(medv_v):
    #I am not sure if i coded a method of bootstrap correctly. I referenced the following websites
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    #https://www.statology.org/bootstrapping-in-python/
    #https://medium.com/swlh/bootstrap-sampling-using-pythons-numpy-85822d868977

    nbootstrap = 1000
    bootstrappercen = []
    for _ in range(nbootstrap):
        bootstrap_sample = random.choice(medv_v, size=len(medv_v), replace=True)
        bootstrap_sample_percen = percentile(bootstrap_sample, 10)
        bootstrappercen.append(bootstrap_sample_percen)

    standard_error = std(bootstrappercen)

    print("Standard Error of mu hat 0.1 using Bootstrap:", standard_error)

def muhat_median(data_a):
    medv_v = data_a[:,13]
    muhatmed = median(medv_v)
    return muhatmed
def main():
    data_a = data_loader("Boston.csv")
    medv_v, muhat = pop_mean(data_a)
    stand_error_muhat(medv_v, muhat)
    standard_error = newboostrapstderror(medv_v)
    standard_error_median = newboostrapstderror_median(medv_v)
    print("standard error of median", standard_error_median)
    print(f'Con Int: [{muhat - 2*standard_error}, {muhat + 2*standard_error}')
    muhatmed = muhat_median(data_a)
    print(muhatmed)
    tenth_percen = percentile(medv_v, 10)
    print("Tenth Percentile (µ̂0.1) of medv:", tenth_percen)
    standard_error_percentile = newbootstrapstderror_tenpercen(medv_v)
    print(standard_error_percentile)
if __name__ == '__main__':
    main()