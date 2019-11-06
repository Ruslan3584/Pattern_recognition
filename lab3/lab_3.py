#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


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


# In[3]:


Y = [zero, one, two, three, four, five, six, seven, eight, nine] # еталонні зображення


# In[4]:


n = Y[0].shape[0]
m = Y[0].shape[1]


# In[5]:


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


# In[6]:


upscale(Y[1],2,2)


# In[7]:


histogram = np.random.randint(0,100,10)   # change value of seed if you want to get new histogram
histogram = histogram/np.sum(histogram)  # normilizing of histogram
histogram


# In[8]:


t = 3  # 3 цифри


# In[12]:


r = np.random.randint(0,9,t)
x_etalone = Y[np.random.randint(0,9)]
for i in range(t-1):
    x_etalone = np.concatenate((x_etalone,Y[np.random.randint(0,9)]),axis=1)


# In[13]:


x_etalone


# In[14]:


Y[np.random.randint(0,9)]


# In[15]:


ksi = np.random.binomial(size=n*m*t, n=1, p=0.2).reshape((n,m*t))


# In[16]:


ksi


# In[17]:


x = x_etalone^ksi


# In[18]:


x


# In[ ]:




