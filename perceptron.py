# Import Libraries

import numpy as np
import matplotlib.pyplot as plt
import random

C = 60
eta = 0.001

def perceptron (x,y):
    n,d = x.shape
    w = np.zeros(d)
    b = 0
    max_pass = 50000
    for t in range(max_pass):
        for i in range(n):
            if y[i] * (np.dot(w.T,x[i,:].T) + b) <= 0:
                w = w + y[i]*x[i,:]
                b = b + y[i]                
    return w,b
