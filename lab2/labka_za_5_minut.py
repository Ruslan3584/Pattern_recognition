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

# In[74]:


import numpy as np


# In[150]:


n = 3 #int(input("Input n: "))
histogram = np.random.randint(0,100,n)
histogram = histogram/np.sum(histogram)


# In[ ]:





# 2. Написать генератор чисел из определённого пользователем набора из n чисел и соответствующей гистограммы—генерируетечислоиз [0;1),считаете cumulative sum для гистограммы,ивыбираете первую из тех ячеек гистограммы, значение в которой больше сгенерированного числа

# In[151]:


g = np.random.uniform(0,1)
cum_hist = np.cumsum(histogram, axis=0)
for i in range(1,len(histogram)):                             
        if cum_hist[i] >= g:
            print(i-1)
            break


# 3. Реализовать функции, которые даёт лучший ответ по заданной гистограмме для каждой из трёх стратегий методом полного перебора: для квадратичного штрафа, для бинарного штрафа, и для стратегии, которую рассмотрели в аудитории (не забудьте, что она требует параметр, который в аудитории назвали α)
# 

# ### Квадратична w(k, k') = (k - k')^2

# In[152]:


r = []
s = []
for i in range(len(histogram)):
    for k in range(len(histogram)):
        r.append(histogram[k]*(k-i)**2)
    s.append(np.sum(r))
    r = []
print(np.argmin(s))


# ### Бінарна w(k, k') = 1(k != k')

# In[153]:


def indicator(i,j):
    return i == j


# In[154]:


r = []
s = []
for i in range(len(histogram)):
    for k in range(len(histogram)):
        r.append(histogram[k]*indicator(k,i))
    s.append(np.sum(r))
    r = []
print(np.argmin(s))


# ### И для стратегии, которую рассмотрели в аудитории (не забудьте, что она требует параметр, который в аудитории назвали α)

# In[163]:


alpha = 0
r = []
s = []
for i in range(len(histogram)):
    for k in range(len(histogram)):
        r.append(histogram[k]*(k-i)**2)
    s.append(np.sum(r)- alpha*histogram[i])
    r = []
print(np.argmin(s))


# In[ ]:





# In[ ]:




