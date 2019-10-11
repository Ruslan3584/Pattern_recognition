import numpy as np


def partial_sum(cum_a, m, n):
    '''

    returns partial sum of array from m to n
    function receives three parameters: array, start-index, end-index
    and calculate the sum of sub-array with interval from 'm' to 'n'

    examples:
    >>> partial_sum([0,1,3,6,10,15],2,4)
    5
    >>> partial_sum([0,1,3,6,10,15],0,3)
    3
    >>> partial_sum([0,1,3,6,10,15],0,6)
    15
    >>> partial_sum([0,1,3,6,10,15],2,1)
    Traceback (most recent call last):
    ...
        raise Exception("ERROR, wrong parameters: m >= n ")
    Exception: ERROR, wrong parameters: m >= n 

    '''

    if m >= n:
        raise Exception("ERROR, wrong parameters: m >= n ")
    z = np.zeros(len(cum_a)+1)
    z[1:] = cum_a
    cum_a = z

    # print("cumsum \n", cum_a)
    # Inclusion-Exclusion principle for n = 1 dimension
    s0 = cum_a[n]
    s1 = -cum_a[m]

    s = int(s0 + s1)
    return s


def summed_area_table(cum_a, m, n, i, j):
    '''

    returns partial area of array[m:n,i:j]
    function receives five  parameters: array, start-row, end-row, start- column, end-column
    calculated cumsum of array for each axis. Then it adds zero column and row to the top left corner
    to avoid checking "if-s"

    examples:
    >>> summed_area_table(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]),1,3,1,3)
    25
    >>> summed_area_table(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]),0,3,0,3)
    41
    >>> summed_area_table(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]),0,3,1,3)
    30
    >>> summed_area_table(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]),0,3,3,1)
    Traceback (most recent call last):
    ...
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")
    Exception: ERROR, wrong parameters: m >= n or i >= j 

    '''

    
    # checking needed parameters for running function
    if m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")
    # calculating cumsum for two axis

    zeros = np.zeros((cum_a.shape[0] + 1, cum_a.shape[1] + 1), dtype="int64")
    zeros[1:, 1:] = cum_a
    cum_a = zeros

    # print("cumsum\n", cum_a)
    # Inclusion-Exclusion principle for n = 2 dimensions
    so = cum_a[n, j]
    s1 = -cum_a[n, i]
    s2 = -cum_a[m, j]
    s3 = cum_a[m, i]

    s = so + s1 + s2 + s3
    return s


def summed_volume_table(cum_a, l, k, m, n, i, j):
    '''
    returns partial area of array[l:k,m:n,i:j]

    returns partial area of array[m:n,i:j]
    function receives seven  parameters: array, (x0, y0 ,z0, x,y,z) - limits of 3D figure
    calculated cumsum of array for each axis. Then it adds zero column and row to the top left corner
    to avoid checking "if-s"

    examples:
    >>> summed_volume_table(np.array([[[1,3,6,10],[3,8,14,21],[8,19,28,40]],[[8,18,30,39],[14,34,53,70],[22,49,71,94]],[[11,28,49,66],[18,49,79,109],[27,72,109,151]]]),0,3,0,3,1,4)
    124
    >>> summed_volume_table(np.array([[[1,3,6,10],[3,8,14,21],[8,19,28,40]],[[8,18,30,39],[14,34,53,70],[22,49,71,94]],[[11,28,49,66],[18,49,79,109],[27,72,109,151]]]))
    Traceback (most recent call last):
    ...
    TypeError: summed_volume_table() missing 6 required positional arguments: 'l', 'k', 'm', 'n', 'i', and 'j'

    '''

    if l >= k or m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: l >=k or m >= n or i >= j ")
    # calculating cumsum for three axis
    
    zeros = np.zeros((cum_a.shape[0] + 1, cum_a.shape[1] + 1, cum_a.shape[2] + 1), dtype="int64")
    zeros[1:, 1:, 1:] = cum_a
    cum_a = zeros
    
    # print("cumsum\n", cum_a, '\n')
    # Inclusion-Exclusion principle for n = 3 dimensions
    s0 =  cum_a[k, n, j]
    s1 = -cum_a[k, n, i]
    s2 = -cum_a[k, m, j]
    s3 = -cum_a[l, n, j]
    s4 =  cum_a[l, m, j]
    s5 =  cum_a[l, n, i]
    s6 =  cum_a[k, m, i]
    s7 = -cum_a[l, m, i]
    
    s = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
    return s



example_1 = np.random.randint(1, 50, (100))
cum_example_1 = example_1.cumsum(axis=0)

example_2 = np.random.randint(1, 50, (7,7))
cum_example_2 = example_2.cumsum(axis=0).cumsum(axis=1)

example_3 = np.random.randint(1, 50, (5, 7, 8)) 
cum_example_3 = example_3.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)


one_dim = partial_sum(cum_example_1, 2, 50)
print(one_dim)

two_dim = summed_area_table(cum_example_2, 0, 5, 1, 3)
print(two_dim)

three_dim = summed_volume_table(cum_example_3,0,5,0,5,1,4)
print(three_dim)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
