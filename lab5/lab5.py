import mnist 
import numpy as np
import time 
import matplotlib.pyplot as plt

#Доров отец

sample= np.load('tmp.npy')  #завантаження бінарізованого датасету

def display(imageAsArray):   # функція  для показу числа
    plt.imshow(imageAsArray, cmap='gray')
    plt.show()



def apriori_probab(array):   # масив із апріорних ймовірностей
    return (1/len(sample))*np.array([np.sum(array[:,0]),np.sum(array[:,1])])


def pixels(array,apr_p): #  повертає пару масивів Pk,ij k=0 or k=1
    h0 = np.empty((28,28))
    h1 = np.empty((28,28))
    k_zeros = array[:,0]
    k_ones = array[:,1]
    
    znam0 = apr_p[0]*len(sample)
    znam1 = apr_p[1]*len(sample)
    h0 = (1/znam0)*np.array([np.sum( sample[:,i][:,j]*k_zeros) for i in range(28) for j in range(28)]).reshape(28,28)
    h1 = (1/znam1)*np.array([np.sum( sample[:,i][:,j]*k_ones) for i in range(28) for j in range(28)]).reshape(28,28)
    return [h0,h1]

def cond_prob_k(h):  # P(Xz | k)
    h0 = h[0]
    h1 = h[1]
    return np.array([[np.prod((h0**sample[z])*((1-h0)**(1 - sample[z]))),np.prod((h1**sample[z])*((1-h1)**(1 - sample[z])))] for z in range(len(sample))]).reshape(len(sample),2)


def result(w,apr_prob): # P(k | Xz)
    pk0 = apr_prob[0]
    pk1 = apr_prob[1]
    b = [w[:,0]*pk0,w[:,1]*pk1]/(w[:,0]*pk0 + w[:,1]*pk1)
    t = np.array([[b[0][i],b[1][i]] for i in range(len(sample))])
    return t


s = [np.random.uniform(low=0.0,high=1,size=2) for i in range(len(sample))]  # початковий розподіл
cond_prob = np.array([s[i]/np.sum(s[i]) for i in range(len(sample))]).reshape(len(sample),2)


# алгоритм
t = 10
for i in range(t):
    apriori_p = apriori_probab(cond_prob)
    P = pixels(cond_prob,apriori_p)
    re = cond_prob_k(P)
    cond_prob = result(re,apriori_p)


print(cond_prob)
