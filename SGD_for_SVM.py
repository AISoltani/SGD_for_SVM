# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import random


def SGD_for_SVM (x,y,eta,C):
    n,d = x.shape
    w = np.zeros(d)
    b = 0
    max_pass = 50000
    
    for t in range(max_pass):
        for i in range(n):
            if y[i] * (np.dot(w.T,x[i,:].T) + b) <= 1:
                w = w - eta *(-2 * C * (1 - y[i] * (np.dot(w.T,x[i,:].T) + b)) * y[i] * x[i,:].T)
                b = b - eta *(-2 * C * (1 - y[i] * (np.dot(w.T,x[i,:].T) + b)) * y[i])    
        w = 1/(1+eta) * w
    return w,b
