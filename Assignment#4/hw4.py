import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import scipy.spatial


def linear_kernel(W , X) :
    return np.matmul(W , X.T)

def RBF_kernel(W , X , sigma) :
    
    dis = scipy.spatial.distance.cdist(W , X , "sqeuclidean")
    return np.exp(- (dis) / (2 * sigma**2))

def polynomial_kernel(W , X , offset , degree) :
    return (offset + np.matmul(W , X.T)) ** degree


def Experiment1() :
    plot_step = 0.01
    points_x = np.arange(-5.0 , 6 , plot_step).reshape(-1 , 1)
    prototypes = np.array([4 , -1 , 0 , 2]).reshape(-1 , 1)
    y = linear_kernel(prototypes , points_x)
    #y = RBF_kernel(prototypes , points_x , sigma = 2)
    #y = polynomial_kernel(prototypes , points_x , offset = 1 , degree = 3)

    for i in range(len(prototypes)) :
        label = "Linear@" + str(prototypes[i])
        plt.plot(points_x , y[i] , label = label)
    plt.legend()
    plt.show()

def main() :
    Experiment1()

if __name__ == "__main__" :
    main()