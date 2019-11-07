#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np


# In[62]:


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


# In[63]:


Y = [zero, one, two, three, four, five, six, seven, eight, nine] # еталонні зображення


# In[64]:


n = Y[0].shape[0]
m = Y[0].shape[1]


# In[65]:


def upscale(orig_array,dig_1, dig_2):
    d = np.array([])
    for i in range(0,orig_array.shape[0]):
        for j in range(0,orig_array.shape[1]):
            if j == 0:
                c  = np.full((dig_1, dig_2),  orig_array[i][j])
            else:
                c = np.concatenate((c,  np.full((dig_1, dig_2),  orig_array[i][j]   )) , axis = 1)
        d = np.append(d,c.reshape(1,c.shape[0]*c.shape[1]))
    d = d.reshape(dig_1*5,dig_2*3)
    d = d.astype('int64') 
    return d


# In[66]:


histogram = np.random.randint(0,100,10)   # change value of seed if you want to get new histogram
histogram = histogram/np.sum(histogram)  # normilizing of histogram
histogram = [0]*10
histogram[1] =1
histogram


# In[74]:


def task(Y, t, p):
    r = np.random.randint(0,9,t)
    x_etalone = Y[r[0]]
    for i in range(1,t):
        x_etalone = np.concatenate((x_etalone,Y[r[i]]),axis=1)
    n = Y[0].shape[0]
    m = Y[0].shape[1]
    ksi = np.random.binomial(size=n*m*t, n=1, p=p).reshape((n,m*t))
    print(r)
    return x_etalone^ksi
x = task(Y,3,0.8)


# In[68]:


x


# In[69]:


def recognition(task,noise_level, numbers):        
    result = []
    for i in numbers:
        d = []
        noise = i^task          
        
        noise = noise.reshape(1, noise.shape[0]*noise.shape[1])
        
        for j in range(0, noise.shape[0]*noise.shape[1]):           
            if noise[0][j] == 0:
                d.append(1-noise_level)
            else:
                d.append(noise_level)

        result.append(np.prod(d))              
    return result


# In[70]:


d = np.hsplit(x, 3)
d


# In[ ]:





# In[71]:


x1 =d[0]
recognition(x1,0.2, Y)*histogram


# In[ ]:





# In[72]:


x12 = np.concatenate((d[0],d[1]),axis=1)


# In[75]:


res = []
for i in range(0,19):
    k2 = []
    for j in range (0,10):
        
        if j>=i-9 and j<=i:
            
            x1 = d[0]
            x2 = d[1]
            p22 = recognition(x2,0.8,Y)[j]*histogram[j]
            p12 = recognition(x1,0.8,Y)[i-j]*histogram[i-j]
            p = p12*p22
            #print(p)
            k2.append(p)
    res.append(np.sum(k2))
(res)


# In[18]:


k2


# In[16]:


recognition(x2,0.2,Y)[2]


# In[ ]:





# In[ ]:





# In[ ]:




