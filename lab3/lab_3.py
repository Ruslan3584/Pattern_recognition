#!/usr/bin/env python
# coding: utf-8

# In[181]:


import numpy as np
import functools


# In[182]:


zero = np.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 0, 1],
                 [1, 0, 1],
                 [1, 1, 1]])

one  = np.array([[0, 1, 0],
                 [0, 1, 0],
                 [0, 1, 0],
                 [0, 1, 0],
                 [0, 1, 0]])

two  = np.array([[1, 1, 1],
                 [0, 0, 1],
                 [1, 1, 1],
                 [1, 0, 0],
                 [1, 1, 1]])

three =np.array([[1, 1, 1],
                 [0, 0, 1],
                 [1, 1, 1],
                 [0, 0, 1],
                 [1, 1, 1]])

four = np.array([[1, 0, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 0, 1],
                 [0, 0, 1]])

five = np.array([[1, 1, 1],
                 [1, 0, 0],
                 [1, 1, 1],
                 [0, 0, 1],
                 [1, 1, 1]])

six =  np.array([[1, 1, 1],
                 [1, 0, 0],
                 [1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]])

seven =np.array([[1, 1, 1],
                 [0, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0],
                 [1, 0, 0]])

eight =np.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1]])


nine = np.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 0, 1],
                 [0, 0, 1]])


# In[183]:


Y = [zero, one, two, three, four, five, six, seven, eight, nine] # еталонні зображення


# In[184]:


n = Y[0].shape[0]
m = Y[0].shape[1]


# In[185]:


histogram = np.random.randint(0,100,10)   # change value of seed if you want to get new histogram
histogram = histogram/np.sum(histogram)  # normilizing of histogram


histogram


# In[186]:


t=20
def task(Y, t, p):
    r = np.random.randint(0,9,t)
    x_etalone = Y[r[0]]
    for i in range(1,t):
        x_etalone = np.concatenate((x_etalone,Y[r[i]]),axis=1)
    n = Y[0].shape[0]
    m = Y[0].shape[1]
    ksi = np.random.binomial(size=n*m*t, n=1, p=p).reshape((n,m*t))
    print(sum(r))
    return x_etalone^ksi
x = task(Y,t,0)


# In[ ]:





# In[187]:


def recognition(task,noise_level, numbers):        
    result = []
    for i in numbers:
        d = np.empty((1,15))[0]
        noise = i^task          
        
        noise = noise.reshape(1, noise.shape[0]*noise.shape[1])
        
        for j in range(0, noise.shape[0]*noise.shape[1]):           
            if noise[0][j] == 0:
                d[j] = 1-noise_level
            else:
                d[j] = noise_level
        #print(len(d))
        result.append(np.prod(d))              
    return result


# In[ ]:





# In[ ]:





# In[ ]:





# In[188]:


im = np.hsplit(x, t)
@functools.lru_cache(None)
def f(t):
    if t==1:
        return recognition(im[t-1],0,Y)*histogram
    else:
        res = []
    for d in range(0,9*t+1):
        k2 = []
        for k in range (0,10):
        
            if k>=d-9*(t-1):
                x2 = im[t-1]
                p22 = recognition(x2,0,Y)[k]*histogram[k]
                k2.append( f(t-1)[d-k]*p22)
        res.append(np.sum(k2))
    #print(t)
    return res


# In[189]:


np.argmax(f(t))

