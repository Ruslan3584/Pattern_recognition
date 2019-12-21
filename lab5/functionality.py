import mnist 
import numpy as np
import time 
import matplotlib.pyplot as plt

#Доров отец

sample= np.load('lab5/tmp.npy')  # завантаження бінарізованого датасету

def display(imageAsArray): 
    '''
    функція  для показу числа
    args: imageAsArray - quadratic numpy array 

    examples:
    >>> display(np.array([[1,2,3],[1,2,3]]))
    Traceback (most recent call last):
    ...
    Exception: image shape is not quadratic
    '''
    if imageAsArray.shape[0] != imageAsArray.shape[1]:
        raise Exception('image shape is not quadratic')
    plt.imshow(imageAsArray, cmap='gray')
    plt.show()


def apriori_probab(array,size):
    '''
    повертає масив із апріорних ймовірностей
    args: array - probabilities P(k | Xz)

    examples:
    >>> apriori_probab(np.array([[0.6, 0.4],[0.05, 0.95],[0.61, 0.39]]),3)
    array([0.42, 0.58])
    >>> apriori_probab(np.array([[0.5, 0.5],[0.5, 0.5]]),2)
    array([0.5, 0.5])
    >>> apriori_probab(np.array([[0.5, 0.5],[0.5, 0.5]]),1)
    Traceback (most recent call last):
    ...
    Exception: wrong size
    '''
    if size <=1:raise Exception('wrong size')
    return (1/size)*np.array([np.sum(array[:,0]),np.sum(array[:,1])])


def pixels(array,apr_p):
    '''
    args: array - probabilities P(k | Xz), apr_p - a priori probabilities P(k)
    returns  two arrays 28x28
    '''
    h0 = np.empty((28,28))
    h1 = np.empty((28,28))
    k_zeros = array[:,0]
    k_ones = array[:,1]
    
    znam0 = apr_p[0]*len(sample)
    znam1 = apr_p[1]*len(sample)
    h0 = (1/znam0)*np.array(
        [np.sum( sample[:,i][:,j]*k_zeros) for i in range(28) for j in range(28)]
        ).reshape(28,28)
    h1 = (1/znam1)*np.array(
        [np.sum( sample[:,i][:,j]*k_ones) for i in range(28) for j in range(28)]
        ).reshape(28,28)
    return [h0,h1]


def cond_prob_k(h):
    '''
    args: h - list of two arrays 28x28
    returns  probabilities P(Xz | k)

    examples:
    >>> cond_prob_k([np.array([[1,2],[3,5]])])
    Traceback (most recent call last):
    ...
    Exception: wrong quantity of arrays
    '''
    if len(h) != 2: raise Exception('wrong quantity of arrays')
    h0 = h[0]
    h1 = h[1]
    return np.array([
        [
            np.prod(
                    (h0**sample[z])*((1-h0)**(1 - sample[z]))
                    ),
            np.prod(
                    (h1**sample[z])*((1-h1)**(1 - sample[z])))
        ] 
        for z in range(len(sample))]
        ).reshape(len(sample),2)


def result(w,apr_prob):
    '''
    args: w - P(Xz | k), apr_prob - P(k)
    returns  probabilities P(k | Xz)
    '''
    pk0 = apr_prob[0]
    pk1 = apr_prob[1]
    b = [w[:,0]*pk0,w[:,1]*pk1]/(w[:,0]*pk0 + w[:,1]*pk1)
    t = np.array([[b[0][i],b[1][i]] for i in range(len(sample))])
    return t

def EM_algorithm(t):
    '''
    args: t - number of iterations
    returns  probabilities P(k | Xz)

    examples:
    >>> EM_algorithm(0)
    Traceback (most recent call last):
    ...
    Exception: number of iterations less than 1
    '''
    if t < 1: raise Exception('number of iterations less than 1')
    s = [np.random.uniform(low=0.0,high=1,size=2) for i in range(len(sample))]  # початковий розподіл
    cond_prob = np.array(
                         [s[i]/np.sum(s[i]) for i in range(len(sample))]
                        ).reshape(len(sample),2)
    # алгоритм
    for i in range(t):
        apriori_p = apriori_probab(cond_prob, len(sample))
        P = pixels(cond_prob,apriori_p)
        re = cond_prob_k(P)
        cond_prob = result(re,apriori_p)
    return cond_prob


if __name__ == "__main__":
    import doctest
    doctest.testmod()
