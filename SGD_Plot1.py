import numpy as np
import matplotlib.pyplot as plt
import random
from SGD_for_SVM import SGD_for_SVM
from perceptron import perceptron 
x = np.array([[1,2],[2,1],[3,1],[3,2]])
y = np.array([[1],[1],[-1],[-1]])
eta = 0.001
C = 10 


w,b = SGD_for_SVM(x, y, eta, C)
wp,bp = perceptron(x, y)
XX1=  np.linspace(0,10,100)
XX2 = (-w[0] * XX1 - b)/w[1]
XX2p = (-wp[0] * XX1 - bp)/wp[1]

plt.figure()
plt.plot(XX1,XX2,label = 'SVM , C=10')
plt.plot(XX1,XX2p,label = 'perceptron')

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.xlim(0,5)
plt.ylim(0,5)
plt.legend()
plt.plot(x[0,0],x[0,1],'b+',markersize=10)
plt.plot(x[1,0],x[1,1],'b+',markersize=10)
plt.plot(x[2,0],x[2,1],'r*',markersize=10)
plt.plot(x[3,0],x[3,1],'r*',markersize=10)
plt.show()
