import numpy as np
import matplotlib.pyplot as plt
import random
from SGD_for_SVM_H import SGD_for_SVM_H

x = np.array([[2,1],[1,2],[3,1],[3,2]])
y = np.array([[1],[1],[-1],[-1]])
eta = 0.001
c = [1,5,10]
plt.figure()
for C  in c:
    w,b = SGD_for_SVM_H (x, y, eta, C)
    XX1=  np.linspace(0,10,100)
    XX2 = (-w[0] * XX1 - b)/w[1]
    ww = w/(w[0])
    bb =b/(w[0])
    print ('C =',C,'w1 =',ww[0],'w2 =',ww[1], 'b =', bb[0])
    plt.plot(XX1,XX2,label = 'C ={}'.format(C),)
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
