#!/usr/bin/env python
# coding: utf-8

#  1. Написать генератор гистограмм на n значений (число n определяет пользователь) — просто генерируете n чисел и делите каждую ячейку на их исходную сумму
# 
# 2. Написать генератор чисел из определённого пользователем набора из n чисел и соответствующей гистограммы—генерируетечислоиз [0;1),считаетеcumulativesumдлягистограммы,ивыбираете первую из тех ячеек гистограммы, значение в которой больше сгенерированного числа 
# 
# 
# 3. Реализовать функции, которые даёт лучший ответ по заданной гистограмме для каждой из трёх стратегий методом полного перебора: для квадратичного штрафа, для бинарного штрафа, и для стратегии, которую рассмотрели в аудитории (не забудьте, что она требует параметр, который в аудитории назвали α) 
# 
# 
# 4. Реализовать функции, которые считают приблизительный риск для двух функций потерь (бинарная и квадратичная) 
# 
#   + на вход подаётся набор чисел и лучший ответ от какой-либо стратегии 
#   + внутри генерируется несколько миллионов или миллиардов чисел из набора согласно гистограмме
#   + считаются штрафы, умножаются на соответствующие им вероятности, и складываются
#   + на выход даётся полученное значение 
# 
# 
# 5. Запустить функцию приблизительного подсчёта риска на разных гистограммах,а результат усреднить (считаем, что каждая гистограмма равновероятна) 
# 
# 
# 6. Посмотреть, при каких значениях параметра α выведенная стратегия работает как одна из ранее изученных детерминированных 
# 
# 
# 7. Понаблюдать, как ведут себя риски бинарной и квадратичной функции потерь при параметрах α, которые находятся между найденными в предыдущем пункте значениями. Объяснить результаты 
# 
# 
# 8. Посмотреть, что будет, если α будет отрицательным. Объяснить результат
# 

# --------------------------
# 1. Написать генератор гистограмм на n значений (число n определяет пользователь) — просто генерируете n чисел и делите каждую ячейку на их исходную сумму

# In[1]:


import numpy as np
import progressbar
import matplotlib.pyplot as plt


# In[58]:


n = 10 #int(input("Input n: "))
K = np.array([3,5,6,8,10,2,0,4,7,9])


# In[3]:


"""
genereting K-set by user 
"""
K = input("Input K: ").split(" ")
if  len(K) !=n:
    raise Exception("ERROR: length K != n")
K = np.array([int(i) for i in K])


# In[59]:


# np.random.seed(135)               # change value of seed if you want to get new histogram
histogram = np.random.randint(0,100,n)
histogram = histogram/np.sum(histogram)  # normilizing of histogram


# In[60]:


histogram


# 2. Написать генератор чисел из определённого пользователем набора из n чисел и соответствующей гистограммы—генерируете число из [0;1),считаете cumulative sum для гистограммы,ивыбираете первую из тех ячеек гистограммы, значение в которой больше сгенерированного числа

# In[61]:


"""
'generator' function gets histogram  from user and generate random number [from 0 to 1)
 calculates cumulative sum for histogram and returns the first index at which the value is greater than random number
"""

def generator(histogram):
    g = np.random.uniform(0,1)
    cum_hist = np.cumsum(histogram, axis=0)
    for i in range(1,len(histogram)):                             
            if cum_hist[i] >= g:
                return i-1

generator(histogram)


# 3. Реализовать функции, которые даёт лучший ответ по заданной гистограмме для каждой из трёх стратегий методом полного перебора: для квадратичного штрафа, для бинарного штрафа, и для стратегии, которую рассмотрели в аудитории (не забудьте, что она требует параметр, который в аудитории назвали α)
# 

# ### Квадратична w(k, k') = (k - k')^2

# In[62]:


"""
'quadratic' function gets K-set and histogram. 
This function finds the strategy which minimize the risk with loss fun  (k - k')^2
"""

def quadratic(K,histogram):
    s = np.empty((len(histogram), ))
    for i in range(len(K)):
        s[i] = np.sum(histogram*(K-K[i])**2)  # calculating risk
    return K[np.argmin(s)]  # returns strategy for appropriate loss function


# In[63]:


quadratic(K,histogram)


# ### Бінарна w(k, k') = 1(k != k')

# In[64]:


"""
'indicator' function describes the work of indicator which uses in probability theory
returns True if condition is correct else False 
"""
def indicator(i,j):
    return i != j
"""
'binary' function gets K-set and histogram. 
This function finds the strategy which minimize the risk with loss function 1ndicator(k != k')
"""
def binary(K,histogram):
    s = np.empty((len(histogram), ))
    for i in range(len(K)):
        s[i] = np.sum(histogram*indicator(K,K[i])) # calculating risk
    return K[np.argmin(s)]  # returns strategy for appropriate loss function


# In[65]:


binary(K,histogram)


# ### И для стратегии, которую рассмотрели в аудитории (не забудьте, что она требует параметр, который в аудитории назвали α)
# 
#  Вывести вид стратегии решения задачи минимизации байесовского риска для функции потерь ω (k,k0) = (k−k0)2 при условии, что байесовский риск для функции потерь ω (k,k0) = 1 (k 6= k0) не превышает определённый порог ε. 

# In[66]:


"""
'third' function gets K-set, alpha and histogram. 
alpha is the variable of dual of a given linear program.
This function finds the strategy which minimize the risk with loss function  (k - k')^2 
with condition that risk for loss function 1ndicator(k != k') is not greater than ε.
"""
def third(K,histogram,alpha=0):
    s = np.empty((len(histogram), ))
    for i in range(len(K)):
        s[i] = (np.sum(histogram*(K-K[i])**2) -alpha*histogram[i]) # calculating risk
    return K[np.argmin(s)]  # returns strategy


# In[67]:


third(K,histogram,alpha=0)


# ### 4. Реализовать функции, которые считают приблизительный риск для двух функций потерь (бинарная и квадратичная) 
# 
#   + на вход подаётся набор чисел и лучший ответ от какой-либо стратегии 
#   + внутри генерируется несколько миллионов или миллиардов чисел из набора согласно гистограмме
#   + считаются штрафы, умножаются на соответствующие им вероятности, и складываются
#   + на выход даётся полученное значение 
# 

# In[68]:


#q = binary(K,histogram)

"""
'r_binary' func gets q - strategy of binary loss function, K-set, histogram.
It returns the risk for appropriate strategy.
"""
def r_binary(q,K,histogram):
    return np.sum(histogram*indicator(K,q))  # calculating risk

print(r_binary( binary(K,histogram) ,K,histogram))


# In[69]:


#q = quadratic(K,histogram)
"""
'r_quadratic' func gets q - strategy of quadratic loss function, K-set, histogram.
It returns the risk for appropriate strategy.
"""
def r_quadratic(q,K,histogram):
    return np.sum(histogram*(K-q)**2) # calculating risk

print(r_quadratic( quadratic(K,histogram),K,histogram))


# ### 5. Запустить функцию приблизительного подсчёта риска на разных гистограммах,а результат усреднить (считаем, что каждая гистограмма равновероятна)

# In[70]:


"""
'mean_quad_bin' func gets only K-set
It generates histogram and returns mean for quadratic and binary risks
with appropriate strategies.
"""

def mean_quad_bin(K):
    quadraticf = np.empty((10**4, ))
    binaryf = np.empty((10**4, ))
    
    for i in progressbar.progressbar(range(10**4)):
        
        histogram = np.random.randint(0,100,n)   # generating new histogram
        histogram = histogram/np.sum(histogram)  # norming
        
        q_q = quadratic(K,histogram)             # strategy for quadratic loss func
        q_b = binary(K,histogram)                # strategy for binary loss func
        
        quadraticf[i] = r_quadratic(q_q, K, histogram) # appending to list which includes risks for 'q_q'
        binaryf[i] = r_binary(q_b, K, histogram)       # appending to list which includes risks for 'q_b'
    
    f = (str('quadratic mean ')+ str(np.mean(quadraticf)),str('binary mean ')+ str(np.mean(binaryf))) # mean values
    return f


# In[71]:


print(mean_quad_bin(K))


# ### 6. Посмотреть, при каких значениях параметра α выведенная стратегия работает как одна из ранее изученных детерминированных

# In[72]:


#q = third(K,histogram)

"""
Я НЕ РОЗУМІЮ НАВІЩО ЦЯ ФУНКЦІЯ ПОТРІБНА
"""
def r_third(q,K,histogram,alpha=0):
    return np.sum(histogram*(K-q)**2) - alpha*histogram[q]  # calculating  
r_third(third(K,histogram),K,histogram)


# In[75]:


"""
'third_mean_q' func gets K-set and parameter alpha
Returns mean risk of quadratic loss function. This risk gets strategy from 'third' func. 
"""

def third_mean_q(K,alpha):
    thirdf = np.empty((10**4, ))
    
    for i in range(10**4):
        histogram = np.random.randint(0,100,n)  # generating histogram
        histogram = histogram/np.sum(histogram)
        
        q = third(K,histogram,alpha)  # calculating the strategy for 'third' func
        
        thirdf[i] = r_quadratic(q,K,histogram) # appending to the list which includes risks of quadratic loss with 'third' strategy
    return np.mean(thirdf)


# In[76]:


"""
'third_mean_b' func gets K-set and parameter alpha
Returns mean risk of binary loss function. This risk gets strategy from 'third' func. 
"""

def third_mean_b(K,alpha):
    thirdf = np.empty((10**4, ))
    for i in range(10**4):
        histogram = np.random.randint(0,100,n)  # generating histogram
        histogram = histogram/np.sum(histogram)
        
        q = third(K,histogram,alpha)  # calculating the strategy for 'third' func
        
        thirdf[i] = r_binary(q,K,histogram) # appending to the list which includes risks of binary loss with 'third' strategy
    return np.mean(thirdf)


# In[81]:


alpha = np.linspace(-5,5,15)  # list of alpha values needed for checking quadratic and binary risks behavior.
alpha


# ### 7. Понаблюдать, как ведут себя риски бинарной и квадратичной функции потерь при параметрах α, которые находятся между найденными в предыдущем пункте значениями. Объяснить результаты

# In[82]:


"""
'means' func gets only alpha-list values.
Returns mean binary and quadratic risks for approprate 'third' strategies.
This functins shows behavior  quadratic and binary risks with defferent alpha values.
"""

def means(alpha):
    mean_R_bibary = np.empty( (len(alpha), ) )
    mean_R_quadratic = np.empty( (len(alpha), ) )

    for i in progressbar.progressbar(range(len(alpha))):
        mean_R_bibary[i] = third_mean_b(K,alpha[i])
        mean_R_quadratic[i] = third_mean_q(K,alpha[i])
        
    f = (mean_R_bibary,mean_R_quadratic)
    return f
m = means(alpha)


# In[83]:


print(m[0])
print(m[1])


# In[84]:


"""
'plotting_R_quadratic' func gets alpha-array,R_quadratic is 'mean_R_quadratic' value of 'means' func.
Returns plot which shows how risk changes with different alpha.
"""
def plotting_R_quadratic(alpha,R_quadratic):
    plt.plot(alpha, R_quadratic)
    plt.xlabel('alpha')
    plt.ylabel('R_quadratic')
    plt.show()


# In[85]:


"""
'plotting_R_binary' func gets alpha-array,R_quadratic is 'mean_R_binary' value of 'means' func.
Returns plot which shows how risk changes with different alpha.
"""
def plotting_R_binary(alpha,R_binary):
    plt.plot(alpha,R_binary)
    plt.xlabel('alpha')
    plt.ylabel('R_binary')
    plt.show()


# In[86]:


plotting_R_quadratic(alpha, m[1])
plotting_R_binary(alpha,m[0])


# In[85]:


#Посмотреть, что будет, если α будет отрицательным. Объяснить результат


# In[86]:


a[0] =1


# In[87]:


a


# In[133]:


np.random.seed(0) ; np.random.rand(4)


# In[ ]:




