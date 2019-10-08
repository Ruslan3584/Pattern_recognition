import numpy as np


def partial_sum(a, m, n):
    print("original\n", a)
    print("area\n", a[m:n])
    if m >= n:
        return "ERROR, wrong parameters: m >= n "
    cum_a = a.cumsum(axis=0)
    print("cumsum \n", cum_a)
    s0 = cum_a[n - 1]
    s1 = -cum_a[m - 1]
    print("np.sum = ", np.sum(a[m:n]))
    if m == 0:
        s1 = 0
    s = s0 + s1
    return s


def summed_area_table(a, m, n, i, j):
    print("original\n", a)
    print("area\n", a[m:n, i:j])

    if m >= n or i >= j:
        return "ERROR, wrong parameters: m >= n or i >= j "

    cum_a = a.cumsum(axis=0).cumsum(axis=1)
    print("cumsum\n", cum_a)
    so = cum_a[n - 1, j - 1]
    s1 = -cum_a[n - 1, i - 1]
    s2 = -cum_a[m - 1, j - 1]
    s3 = cum_a[m - 1, i - 1]
    print("np.sum = ", np.sum(a[m:n, i:j]))

    if i == 0 and m == 0:
        s1 = s2 = s3 = 0
    if m == 0:
        s2 = s3 = 0
    if i == 0:
        s1 = s3 = 0
    s = so + s1 + s2 + s3
    return s


def summed_volume_table(a, l, k, m, n, i, j):
    print("original\n", a, '\n')
    print("volume\n", a[l:k, m:n, i:j], '\n')

    if l >= k or m >= n or i >= j:
        return "ERROR, wrong parameters: l >=k or m >= n or i >= j "

    cum_a = a.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    print("cumsum\n", cum_a, '\n')
    s0 = cum_a[k - 1, n - 1, j - 1]
    s1 = -cum_a[k - 1, n - 1, i - 1]
    s2 = -cum_a[k - 1, m - 1, j - 1]
    s3 = -cum_a[l - 1, n - 1, j - 1]
    s4 = cum_a[l - 1, m - 1, j - 1]
    s5 = cum_a[l - 1, n - 1, i - 1]
    s6 = cum_a[k - 1, m - 1, i - 1]
    s7 = -cum_a[l - 1, m - 1, i - 1]
    print("np.sum =", np.sum(a[l:k, m:n, i:j]), '\n')

    if l == 0 and m == 0:
        s2 = s3 = s4 = s5 = s6 = s7 = 0
    if m == 0 and i == 0:
        s1 = s2 = s4 = s5 = s6 = s7 = 0
    if l == 0 and i == 0:
        s1 = s3 = s4 = s4 = s6 = s7 = 0
    if l == 0:
        s3 = s4 = s5 = s7 = 0
    if m == 0:
        s2 = s4 = s6 = s7 = 0
    if i == 0:
        s1 = s5 = s6 = s7 = 0
    s = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
    return s


example_1 = np.array(range(0, 6))

example_2 = np.array([[1, 2, 3, 4,  2],
                      [3, 3, 5, 6,  3],
                      [7, 8, 9, 10, 4],
                      [4, 5, 3, 11, 0]])
example_3 = g = np.array([[1, 2, 3, 4, 2, 3],
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
two_dim = summed_area_table(example_2, 2, 4, 1, 4)
print(two_dim)
three_dim = summed_volume_table(example_3, 0, 3, 0, 3, 1, 4)
print(three_dim)
