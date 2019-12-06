#!/usr/bin/env python
# coding: utf-8

# In[217]:


from PIL import Image
import numpy as np
import functools


# In[218]:


img2 = np.array([0,0,0,1])#np.array(Image.open('examples/im2.ppm').convert('L'))
img6 = np.array([0,0,1,0])#np.array(Image.open('examples/im6.ppm').convert('L'))
img2.shape


# In[219]:


disp = np.array(Image.open('examples/disp2.pgm'))


# In[220]:


disp[0]


# In[221]:


disp.max()


# In[222]:


maxd = 3#int(disp.max())
alpha=1


# In[223]:


maxd


# In[224]:


def h(y,d):
    L = img2[y]#img2[0,y]
    R = img6[y-d]#img6[0,y-d]
    return np.sqrt((L - R)**2)


# In[225]:


def g(d,d1):
    return np.sqrt((d - d1)**2)


# In[226]:


G = np.empty((maxd+1,maxd+1))
for i in range(maxd+1):
    for j in range(maxd+1):
        G[i,j]= g(i,j)


# In[227]:


g(2,1)


# In[228]:


H


# In[229]:


f0 =  np.full((maxd+1),np.inf).astype('float64')
for i in range(0,maxd+1):
    if i < 1:
        #print(h(0,0))
        f0[i] = h(0,0)
f0


# In[230]:


f1 =  np.full((maxd+1),np.inf).astype('float64')
for i in range(0,maxd+1):
    if i < 2:
        f1[i] = min(f0 + G[i,]) + h(1,i) 
f1    


# In[231]:


f2 = np.full((maxd+1),np.inf).astype('float64')
for i in range(0,maxd+1):
    if i < 3:
        f2[i] = min(f1 + G[i,]) + h(2,i) 
f2   


# In[232]:


f3 = np.full((maxd+1),np.inf).astype('float64')
for i in range(0,maxd+1):
    if i < 4:
        f3[i] = min(f2 + G[i,]) + h(3,i) 
f3   


# In[233]:


f2 + g(np.array(range(0,maxd+1)).astype('float64'),0)


# In[234]:


f0 + g(np.array(range(0,maxd+1)).astype('float64'),0)


# In[235]:


g(2,3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[238]:


dm = []
@functools.lru_cache(None)
def r(m):
    print(m)
    if m == 0:
        f0 =  np.full((maxd+1),np.inf).astype('float64')
        f0[0] = h(0,0)
        dm.append(np.argmin(f0))
        return f0
    else:
        q = r(m-1)
        f =  np.full((maxd+1),np.inf).astype('float64')
        for j in range(1,m+1):
            for i in range(0,maxd+1):
                if i < j + 1:
                    f[i] = min(q + G[i,]) + h(j,i)
        dm.append(np.argmin(f+G[i,]))
        return f


# In[ ]:





# In[239]:


dm


# In[ ]:


disp[


# In[242]:


r(3),dm


# In[ ]:





# In[ ]:





# In[243]:


d3 = np.argmin(r(3))
it = np.array(range(0,maxd+1)).astype('float64')
it


# In[244]:


d2 = np.argmin(r(2) + g(it,d3))


# In[245]:


d1 = np.argmin(r(1) + g(it,d2))


# In[246]:


d0 = np.argmin(r(0) + g(it,d1))


# In[247]:


dd = np.array([d0,d1,d2,d3])
dd


# In[249]:


img6 - img2


# In[ ]:





# In[ ]:





# In[26]:


g(it,d3)


# In[27]:


(np.sort(bn) == bn).all()


# In[461]:


res = []
di = np.array(range(0,maxd+1)).astype('float64')
m = 100
for i in reversed(range(0,m+1)):
    print(i)
    if i == m:
        res.append(np.argmin(r(m)))
    else:
        res.append(np.argmin(r(m-1)+ g(di,res[i - m])))
     


# In[ ]:


res


# In[ ]:


for t in reversed(range(0,3)):
    print(t)


# In[462]:


disp[220][:100]


# In[ ]:





# In[24]:


d20 = np.argmin(r(44))
d20
it = np.array(range(0,maxd+1)).astype('float64')
d19 = np.argmin(r(43) + g(it,d20))
d20


# In[21]:


di = np.array(range(0,maxd+1)).astype('float64')


# In[23]:


disp[0][:45]


# In[12]:


f1 = np.array(range(0,maxd+1)).astype('float64')
d1 = np.array(range(0,maxd+1)).astype('float64')
for i in range(0,maxd+1):
    if  i <2:
        s = min(f0 + G[i,])#vg(d1,i)
        f1[i] = s + h(1,i)
    else:
        f = np.array(range(0,maxd+1)).astype('float64')
        for i in range(0,maxd+1):
            if  i <:
                s = min(f0 + G[i,])#vg(d1,i)
                f1[i] = s + h(1,i)
            else:
                f1[i] = np.inf
        return 


# In[38]:


length = img2.shape[0]
result = []
def z(m):
    if m == 0:
        f0 =  np.full((maxd+1),np.inf).astype('float64')
        f0[0] = h(0,0)
        return f0
    else:
        for j in range(450):
            f = np.full((maxd+1),np.inf).astype('float64')
            q = z(m-1)
            f[:m+1] = [min(q + G[i,]) + h(m,i) for i in range(0,m+1)]
            result.append(np.argmin(f[:m+1]))
    return f 


# In[14]:


for i in range(1,1):
    print('v')


# In[ ]:





# In[116]:


g(np.array([1,2]),1)


# In[ ]:





# In[ ]:




