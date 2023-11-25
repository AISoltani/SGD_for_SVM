# Import Libraries...

import numpy as np
import matplotlib.pyplot as plt
import random

def SGD_for_SVM_H (x,y,eta,C):
    n,d = x.shape
    w = np.zeros(d)
    b = 0
    max_pass = 50000
    
    for t in range(max_pass):
        for i in range(n):
            if y[i] * (np.dot(w.T,x[i,:].T) + b) <= 1:
                w = w - eta *(- C * y[i] * x[i,:].T)
                b = b - eta *(- C * y[i])    
        w = 1/(1+eta) * w
    return w,b
