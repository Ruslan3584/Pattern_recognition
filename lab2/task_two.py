import numpy as np
import progressbar
import matplotlib.pyplot as plt
from sys import argv

def set_parameters(size):
    '''
    genereting K-set by user len(K) must be equal size
    '''
    K = input("Input K: ").split(" ")
    if  len(K) != size:
        raise Exception("ERROR: length K != size")
    K = np.array([int(i) for i in K])
    return K

def set_histogram(size,seed=135):
    '''
    create histogrmam len = size
    default seed = 135
   
    examples:
    >>> set_histogram(10,30)
    array([0.11671924, 0.11671924, 0.14195584, 0.14195584, 0.03785489,
           0.07255521, 0.00630915, 0.16719243, 0.05362776, 0.14511041])
    >>> set_histogram(10)
    array([0.06175299, 0.11752988, 0.1812749 , 0.14940239, 0.1374502 ,
           0.11155378, 0.0438247 , 0.12350598, 0.02390438, 0.0498008 ])
    >>> set_histogram(-10,30)
    Traceback (most recent call last):
    ...
    Exception: size <= 0
    '''
    if size <= 0:
    	raise Exception("size <= 0")
    np.random.seed(seed)
    histogram = np.random.randint(0,100,size)   # change value of seed if you want to get new histogram
    histogram = histogram/np.sum(histogram)  # normilizing of histogram
    return histogram



def generator(histogram,uniform):
    '''
    'generator' function gets histogram and random number [from 0 to 1)
    calculates cumulative sum for histogram 
    and returns the first index at which the value is greater than random number ~(Union(0,1])

    examples:
    >>> generator(np.array([0.06175299, 0.11752988, 0.1812749 , 0.14940239, 0.1374502 ,0.11155378, 0.0438247 , 0.12350598, 0.02390438, 0.0498008 ]),0.5)
    2
    >>> generator(np.array([0.06175299, 0.11752988, 0.1812749 , 0.14940239, 0.1374502 ,0.11155378, 0.0438247 , 0.12350598, 0.02390438, 0.0498008 ]),0)
    0
    >>> generator(np.array([0.06175299, 0.11752988, 0.1812749 , 0.14940239, 0.1374502 ,0.11155378, 0.0438247 , 0.12350598, 0.02390438, 0.0498008 ]),-1)
    Traceback (most recent call last):
    ...
    Exception: uniform < 0
    '''
    if uniform < 0:
    	raise Exception("uniform < 0")
    cum_hist = np.cumsum(histogram, axis=0)
    for i in range(1,len(histogram)):                             
            if cum_hist[i] >= uniform:
                return i-1


def quadratic(K,histogram):
    '''
    'quadratic' function gets K-set and histogram. 
    This function finds the strategy which minimize the risk with loss fun  (k - k')^2

    examples:
    >>> quadratic(np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.11671924,0.11671924,0.14195584,0.14195584,0.03785489,0.07255521,0.00630915,0.16719243, 0.05362776, 0.14511041]))
    6
    >>> quadratic(np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.2, 0.072, 0.016, 0.00533333, 0.008,0.17866667,0.20266667,0.128,0.05866667,0.13066667]))
    4
    '''
    s = np.empty((len(histogram), ))
    for i in range(len(K)):
        s[i] = np.sum(histogram*(K-K[i])**2)  # calculating risk
    return K[np.argmin(s)]  # returns strategy for appropriate loss function (k - k')^2



def indicator(i,j):
    '''
    'indicator' function describes the work of indicator which uses in probability theory
    returns True if condition is correct else False 

    examples:
    >>> indicator(1,1)
    False
    >>> indicator(1,2)
    True
    '''
    return i != j



def binary(K,histogram):
    '''
    'binary' function gets K-set and histogram. 
    This function finds the strategy which minimize the risk with loss function 1ndicator(k != k')

    examples:
    >>> binary(np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.2, 0.072, 0.016, 0.00533333, 0.008,0.17866667,0.20266667,0.128,0.05866667,0.13066667]))
    0
    >>> binary(np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.11671924,0.11671924,0.14195584,0.14195584,0.03785489,0.07255521,0.00630915,0.16719243, 0.05362776, 0.14511041]))
    4
    '''
    s = np.empty((len(histogram), ))
    for i in range(len(K)):
        s[i] = np.sum(histogram*indicator(K,K[i])) # calculating risk
    return K[np.argmin(s)]  # returns strategy for appropriate loss function



def third(K,histogram,alpha=0):
    '''
    'third' function gets K-set, alpha and histogram. 
    alpha is the variable of dual of a given linear program.
    This function finds the strategy which minimize the risk with loss function  (k - k')^2 
    with condition that risk for loss function 1ndicator(k != k') is not greater than eps.

    examples:
    >>> third(np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.11671924,0.11671924,0.14195584,0.14195584,0.03785489,0.07255521,0.00630915,0.16719243, 0.05362776, 0.14511041]))
    6
    >>> third(np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.11671924,0.11671924,0.14195584,0.14195584,0.03785489,0.07255521,0.00630915,0.16719243, 0.05362776, 0.14511041]),0)
    6
    '''
    s = np.empty((len(histogram), ))
    for i in range(len(K)):
        s[i] = (np.sum(histogram*(K-K[i])**2) -alpha*histogram[i]) # calculating risk
    return K[np.argmin(s)]  # returns strategy




def r_binary(q,K,histogram):
    '''
    'r_binary' func gets q - strategy of binary loss function, K-set, histogram.
    It returns the risk for appropriate strategy.

    examples:
    >>> r_binary(4,np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.11671924,0.11671924,0.14195584,0.14195584,0.03785489,0.07255521,0.00630915,0.16719243, 0.05362776, 0.14511041]))
    0.83280758
    >>> r_binary(6,np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.11671924,0.11671924,0.14195584,0.14195584,0.03785489,0.07255521,0.00630915,0.16719243, 0.05362776, 0.14511041]))
    0.85804417
    '''
    return np.sum(histogram*indicator(K,q))  # calculating risk




def r_quadratic(q,K,histogram):
    '''
    'r_quadratic' func gets q - strategy of quadratic loss function, K-set, histogram.
    It returns the risk for appropriate strategy.
    examples:
    >>> r_quadratic(4,np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.11671924,0.11671924,0.14195584,0.14195584,0.03785489,0.07255521,0.00630915,0.16719243, 0.05362776, 0.14511041]))
    8.93690865
    >>> r_quadratic(6,np.array([3,5,6,8,10,2,0,4,7,9]),np.array([0.11671924,0.11671924,0.14195584,0.14195584,0.03785489,0.07255521,0.00630915,0.16719243, 0.05362776, 0.14511041]))
    5.7570979300000005
    '''
    return np.sum(histogram*(K-q)**2) # calculating risk


def mean_quad_bin(K,size,N=10**4):
    '''
    'mean_quad_bin' func gets only K-set
    It generates histogram and returns mean for quadratic and binary risks
    with appropriate strategies.
    '''
    quadraticf = np.empty((N, ))
    binaryf = np.empty((N, ))
    #for i in progressbar.progressbar(range(N)):
    for i in range(N):
        

        histogram = set_histogram(size,np.random.seed(np.random.randint(100)))   # generating new histogram
        
        q_q = quadratic(K,histogram)             # strategy for quadratic loss func
        q_b = binary(K,histogram)                # strategy for binary loss func
        
        quadraticf[i] = r_quadratic(q_q, K, histogram) # appending to list which includes risks for 'q_q'
        binaryf[i] = r_binary(q_b, K, histogram)       # appending to list which includes risks for 'q_b'
    
    f = (np.mean(quadraticf),np.mean(binaryf)) # mean values
    return f


def third_mean_q(K,alpha,size,N=10**4):
    '''
    'third_mean_q' func gets K-set , parameter alpha , size of histogram, and number of repeats N
    Returns mean risk of quadratic loss function. This risk gets strategy from 'third' func. 
    '''
    thirdf = np.empty((N, ))
    
    for i in range(N):
        histogram = set_histogram(size,np.random.seed(np.random.randint(100)))  # generating histogram
        
        q = third(K,histogram,alpha)  # calculating the strategy for 'third' func
        
        thirdf[i] = r_quadratic(q,K,histogram) # appending to the list which includes risks of quadratic loss with 'third' strategy
    return np.mean(thirdf)


def third_mean_b(K,alpha,size,N=10**4):
    '''
    'third_mean_b' func gets K-set and parameter alpha
    Returns mean risk of binary loss function. This risk gets strategy from 'third' func. 
    '''
    thirdf = np.empty((N, ))
    for i in range(N):
        histogram = set_histogram(size,np.random.seed(np.random.randint(100)))  # generating histogram
        
        q = third(K,histogram,alpha)  # calculating the strategy for 'third' func
        
        thirdf[i] = r_binary(q,K,histogram) # appending to the list which includes risks of binary loss with 'third' strategy
    return np.mean(thirdf)


def means(alpha):
    '''
    'means' func gets only alpha-list values.
    Returns mean binary and quadratic risks for approprate 'third' strategies.
    This functins shows behavior  quadratic and binary risks with defferent alpha values.
    '''
    mean_R_binary = np.empty( (len(alpha), ) )
    mean_R_quadratic = np.empty( (len(alpha), ) )

    for i in progressbar.progressbar(range(len(alpha))):
    #for i in range(len(alpha)):
        mean_R_binary[i] = third_mean_b(K,alpha[i],n)
        mean_R_quadratic[i] = third_mean_q(K,alpha[i],n)
        
    f = (mean_R_binary,mean_R_quadratic)
    return f


def plotting_R_quadratic(alpha,R_quadratic):
    '''
    'plotting_R_quadratic' func gets alpha-array,R_quadratic is 'mean_R_quadratic' value of 'means' func.
    Returns plot which shows how risk changes with different alpha.
    '''
    plt.plot(alpha, R_quadratic)
    plt.xlabel('alpha')
    plt.ylabel('R_quadratic')
    plt.show()



def plotting_R_binary(alpha,R_binary):
    '''
    'plotting_R_binary' func gets alpha-array,R_quadratic is 'mean_R_binary' value of 'means' func.
    Returns plot which shows how risk changes with different alpha.
    '''
    plt.plot(alpha,R_binary)
    plt.xlabel('alpha')
    plt.ylabel('R_binary')
    plt.show()




n = 10 #int(input("Input n: "))
K = np.array([3,5,6,8,10,2,0,4,7,9])

histogram = set_histogram(n,np.random.seed(np.random.randint(100)))


#n = int(argv[1])
#print("size = ",n)
#K = set_parameters(n)
#histogram = set_histogram(n)

q_quadr = quadratic(K,histogram) # strategy for quadratic loss function 
print("strategy q_quadr: ",q_quadr)
q_bin = binary(K,histogram) # strategy for binary loss function 
print("strategy q_bin: ",q_bin)
q_third = third(K,histogram) # strategy for third loss function alpha = 0
print("strategy q_third: ",q_third)



r_b = r_binary( binary(K,histogram) ,K,histogram) # R(q) for binary 
print("mean R(q) for binary: ",r_b)


r_q = r_quadratic( quadratic(K,histogram),K,histogram) # R(q) for quadratic  
print("mean R(q) for quadratic: ",r_q)


alpha = np.linspace(-50,50,15)  # list of alpha values needed for checking quadratic and binary risks behaviour.

#m = means(alpha) # m = mean_R_binary,mean_R_quadratic

#plotting_R_quadratic(alpha, m[1]) # mean_R_quadratic
#plotting_R_binary(alpha,m[0]) # mean_R_binary
