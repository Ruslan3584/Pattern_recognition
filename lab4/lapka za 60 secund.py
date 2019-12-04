#!/usr/bin/env python
# coding: utf-8

# In[66]:


from PIL import Image
import numpy as np


# In[67]:


img2 = np.array(Image.open('examples/im2.ppm').convert('L'))
img6 =  np.array(Image.open('examples/im6.ppm').convert('L'))
img2.shape


# In[71]:


disp = np.array(Image.open('examples/disp2.pgm'))


# In[82]:


disp.max()


# In[72]:


maxd = disp.max()
alpha=1


# In[73]:


maxd


# In[22]:


def h(y,d):
    L = img2[0,y]
    R = img6[0,y-d]
    return np.sqrt(L**2 + R**2)


# In[23]:


def g(d,d1):
    return np.sqrt(d**2 + d1**2)


# In[24]:


G = np.empty((maxd+1,maxd+1))
for i in range(maxd+1):
    for j in range(maxd+1):
        G[i,j]= g(i,j)


# In[25]:


d1 = np.array(range(0,maxd+1))


# In[26]:


h(0,np.array([0,1]))


# In[27]:


vh = np.vectorize(h)


# In[28]:


vh(0,[0,1])


# In[29]:


vg = np.vectorize(g)


# In[ ]:





# In[30]:


d1 = np.array(range(0,maxd+1))


# In[31]:


f1 = np.array(range(0,maxd+1))
for i in range(0,maxd+1):
    f1[i] = h(0,i)


# In[40]:


f2 = np.array(range(0,maxd+1))
for i in range(0,maxd+1):
    s = min(f1 + vg(d1,i))
    f2[i] = s + h(1,i)
f1


# In[51]:


d1 = np.array(range(0,maxd+1))
result = []
def z(m):
    if m == 1:
        f1 = np.array(range(0,maxd+1))
        for i in range(0,maxd+1):
            f1[i] = h(0,i)
        return f1
    else:
        q = np.array(range(0,maxd+1))
        e = z(m-1)
        for i in range(0,maxd+1):
            s = min(e + vg(d1,i))
            result.append(np.argmin(s))
            q[i] = s + h(m-1,i)
        return q


# In[57]:


z(300)


# In[ ]:





# In[ ]:




