import numpy as np


def partial_sum(a, m, n):
    # print("original\n", a)
    # print("area\n", a[m:n])

    '''

    returns partial sum of array from m to n

    examples:
    >>> partial_sum(np.array(range(0, 6)),2,4)
    5
    >>> partial_sum(np.array(range(0, 6)),0,3)
    3
    >>> partial_sum(np.array(range(0, 6)),2,1)
    Traceback (most recent call last):
    ...
        raise Exception("ERROR, wrong parameters: m >= n ")
    Exception: ERROR, wrong parameters: m >= n 

    '''

    if m >= n:
        raise Exception("ERROR, wrong parameters: m >= n ")
    cum_a = a.cumsum(axis=0)

    cum_a[-1] = 0

    # print("cumsum \n", cum_a)
    s0 = cum_a[n - 1]
    s1 = -cum_a[m - 1]
    # print("np.sum = ", np.sum(a[m:n]))

    s = s0 + s1
    return s


def summed_area_table(a, m, n, i, j):
    # print("original\n", a)
    # print("area\n", a[m:n, i:j])

    '''

    returns partial area of array[m:n,i:j]

    examples:
    >>> summed_area_table(np.array([[1, 2, 3, 4,  2],[3, 3, 5, 6,  3],[7, 8, 9, 10, 4],[4, 5, 3, 11, 0]]),1,3,1,3)
    33
    >>> summed_area_table(np.array([[1, 2, 3, 4,  2],[3, 3, 5, 6,  3],[7, 8, 9, 10, 4],[4, 5, 3, 11, 0]]),0,3,0,3)
    60
    >>> summed_area_table(np.array([[1, 2, 3, 4,  2],[3, 3, 5, 6,  3],[7, 8, 9, 10, 4],[4, 5, 3, 11, 0]]),0,3,1,3)
    44
    >>> summed_area_table(np.array([[1, 2, 3, 4,  2],[3, 3, 5, 6,  3],[7, 8, 9, 10, 4],[4, 5, 3, 11, 0]]),0,3,3,1)
    Traceback (most recent call last):
    ...
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")
    Exception: ERROR, wrong parameters: m >= n or i >= j 

    '''



    if m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")

    cum_a = a.cumsum(axis=0).cumsum(axis=1)
    zeros = np.zeros((cum_a.shape[0] + 1, cum_a.shape[1] + 1), dtype="int64")
    zeros[1:, 1:] = cum_a

    # print("cumsum\n", cum_a)
    so = cum_a[n, j]
    s1 = -cum_a[n, i]
    s2 = -cum_a[m, j]
    s3 = cum_a[m, i]

    # print("np.sum = ", np.sum(a[m:n, i:j]))
    s = so + s1 + s2 + s3
    return s


def summed_volume_table(a, l, k, m, n, i, j):
    # print("original\n", a, '\n')
    # print("volume\n", a[l:k, m:n, i:j], '\n')


    '''
    returns partial area of array[l:k,m:n,i:j]
    examples:
    >>> summed_volume_table(np.array([[1, 2, 3, 4, 2, 3],[3, 3, 5, 6, 3, 5],[7, 8, 9, 5, 4, 7],[4, 5, 3, 1, 0, 1],[3, 7, 9, 8, 1, 4],[2, 5, 1, 7, 4, 6]],ndmin= 3).reshape(3,3,4),0,3,0,3,1,4)
    124
    >>> summed_volume_table(np.array(range(0, 6)))
    Traceback (most recent call last):
    ...
    TypeError: summed_volume_table() missing 6 required positional arguments: 'l', 'k', 'm', 'n', 'i', and 'j'

    '''
    if l >= k or m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: l >=k or m >= n or i >= j ")

    cum_a = a.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)

    zeros = np.zeros((cum_a.shape[0] + 1, cum_a.shape[1] + 1, cum_a.shape[2] + 1), dtype="int64")
    zeros[1:, 1:, 1:] = cum_a
    cum_a = zeros

    # print("cumsum\n", cum_a, '\n')
    s0 =  cum_a[k, n, j]
    s1 = -cum_a[k, n, i]
    s2 = -cum_a[k, m, j]
    s3 = -cum_a[l, n, j]
    s4 =  cum_a[l, m, j]
    s5 =  cum_a[l, n, i]
    s6 =  cum_a[k, m, i]
    s7 = -cum_a[l, m, i]
    # print("np.sum =", np.sum(a[l:k, m:n, i:j]), '\n')

    s = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
    return s


example_1 = np.array(range(0, 6))

example_2 = np.array([[1, 2, 3, 4,  2],
                      [3, 3, 5, 6,  3],
                      [7, 8, 9, 10, 4],
                      [4, 5, 3, 11, 0]])

example_3 = np.array([[1, 2, 3, 4, 2, 3],
                      [3, 3, 5, 6, 3, 5],
                      [7, 8, 9, 5, 4, 7],
                      [4, 5, 3, 1, 0, 1],
                      [3, 7, 9, 8, 1, 4],
                      [2, 5, 1, 7, 4, 6],
                      [5, 1, 0, 1, 3, 7],
                      [8, 4, 3, 2, 5, 6],
                      [2, 3, 5, 6, 8, 1],
                      [4, 6, 7, 1, 4, 9],
                      [0, 2, 7, 8, 1, 5],
                      [2, 3, 4, 6, 4, 2]], ndmin=3).reshape(3, 4, 6)

one_dim = partial_sum(example_1, 2, 4)
print(one_dim)
two_dim = summed_area_table(example_2, 1, 3, 1, 3)
print(two_dim)
three_dim = summed_volume_table(example_3,0,3,0,3,0,3)
print(three_dim)
