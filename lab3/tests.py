import numpy as np

def checking(a):
    '''
    args:  a - list of probabilities, len(a) = 9*t, where t is number of digits
    
    examples:
    >>> checking(np.array([0.08695652, 0.03933747, 0.14285714, 0.12836439, 0.10144928,0.0952381 , 0.13664596, 0.01656315, 0.04968944, 0.20289855]))
    True
    >>> checking(np.array([0.11848341, 0.10900474, 0.1943128 , 0.22037915, 0.06872038,0.08530806, 0.00473934, 0.06635071, 0.09952607, 0.03317536]))
    False
    >>> checking([])
    Traceback (most recent call last):
    ...
    Exception: a is empty
    '''
    if len(a)==0:raise Exception("a is empty")
    k1 = [] # - множина тих k які дільться на 3 з остачею 
    k2 = [] # - множина тих k які дільться на 3 без остачі
    for i in range(len(a)):
        if i%3 == 0:
            k2.append(a[i])  
        if i%3 != 0:
            k1.append(a[i])

    if sum(k2) >= sum(k1):  # стратегія: в залежності від сум приймаємо рішення
        return True
    else:
        return False



def generate_histogram():
    '''
    create histogrmam len = 10
   
    examples:
    >>> np.random.seed(30)
    >>> generate_histogram()
    array([0.11671924, 0.11671924, 0.14195584, 0.14195584, 0.03785489,
           0.07255521, 0.00630915, 0.16719243, 0.05362776, 0.14511041])
    >>> np.random.seed(5)
    >>> generate_histogram()
    array([0.18539326, 0.14606742, 0.11423221, 0.02996255, 0.13670412,
           0.01498127, 0.11610487, 0.0505618 , 0.05617978, 0.14981273])
    '''
    #np.random.seed(n_seed)
    histogram = np.random.randint(0,100,10)   # change value of seed if you want to get new histogram
    histogram = histogram/np.sum(histogram)  # normilizing of histogram
    return histogram



def upscale(array,  scale_1, scale_2):
    '''
    збільшуємо зображення в потрібну кількість разів

    examples:
    >>> upscale(np.arange(10),2,-2)
    Traceback (most recent call last):
    ...
    Exception: wrong scales
    >>> upscale(np.arange(10),-2,2)
    Traceback (most recent call last):
    ...
    Exception: wrong scales
    >>> upscale(np.arange(10),2,2)
    Traceback (most recent call last):
    ...
    Exception: wrong dimension
    >>> upscale(np.array([[1,2],[2,4]]), 2,2)
    array([[1, 1, 2, 2],
           [1, 1, 2, 2],
           [2, 2, 4, 4],
           [2, 2, 4, 4]])
    >>> upscale(np.array([[1,2,3],[2,4,8]]), 1,1)
    array([[1, 2, 3],
           [2, 4, 8]])
    '''
    if scale_1<=0 or scale_2<=0:
        raise Exception("wrong scales")
    elif len(array.shape)==1:
        raise Exception("wrong dimension")
    else:
        return array.repeat(scale_1,axis=0).repeat(scale_2, axis=1).astype('int64')



if __name__ == "__main__":
    import doctest
    doctest.testmod()
