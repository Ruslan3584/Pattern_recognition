import numpy as np


def partial_sum(cum_a, m, n):
    '''
    returns partial sum of array from m to n
    function receives three parameters: cumulatived sum - array, start-index, end-index
    calculated the Inclusion-Exclusion principle

    examples:
    >>> partial_sum(adding_zeros(np.array([0,1,3,6,10,15]).cumsum(axis=0)),2,4)
    9
    >>> partial_sum(adding_zeros(np.array([0,1,3,6,10,15]).cumsum(axis=0)),0,3)
    4
    >>> partial_sum(adding_zeros(np.array([0,1,3,6,10,15]).cumsum(axis=0)),0,6)
    35
    >>> partial_sum(adding_zeros(np.array([0,1,3,6,10,15]).cumsum(axis=0)),2,1)
    Traceback (most recent call last):
    ...
        raise Exception("ERROR, wrong parameters: m >= n ")
    Exception: ERROR, wrong parameters: m >= n 
    '''

    if m >= n:
        raise Exception("ERROR, wrong parameters: m >= n ")
    # print("cumsum \n", cum_a)
    # Inclusion-Exclusion principle for n = 1 dimension
    s0 = cum_a[n]
    s1 = -cum_a[m]

    return s0 + s1


def simple_partial_sum(array,m,n):
    '''
    calculate partial_sum using standart np.sum() function 
    examples:
    >>> simple_partial_sum(np.array([0,1,3,6,10,15]),0,6)
    35
    >>> simple_partial_sum(np.array([0,1,3,6,10,15]),2,6)
    34
    >>> simple_partial_sum(np.array([0,1,3,6,10,15]),3,0)
    Traceback (most recent call last):
    ...
        raise Exception("ERROR, wrong parameters: m >= n ")
    Exception: ERROR, wrong parameters: m >= n 
    '''
    if m >= n:
        raise Exception("ERROR, wrong parameters: m >= n ")
    # made needed slice from array and summed it
    res = np.sum(array[m:n])
    return res



def summed_area_table(cum_a, m, n, i, j):
    '''
    returns summed area table of array[m:n,i:j]
    function receives five  parameters:  cumulatived sum-array, start-row, end-row, start- column, end-column
    calculated the Inclusion-Exclusion principle

    examples:
    >>> summed_area_table(adding_zeros(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]).cumsum(axis=0).cumsum(axis=1)),1,3,1,3)
    91
    >>> summed_area_table(adding_zeros(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]).cumsum(axis=0).cumsum(axis=1)),0,3,0,3)
    116
    >>> summed_area_table(adding_zeros(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]).cumsum(axis=0).cumsum(axis=1)),0,4,0,5)
    606
    >>> summed_area_table(adding_zeros(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]).cumsum(axis=0).cumsum(axis=1)),0,3,1,3)
    100
    >>> summed_area_table(adding_zeros(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]).cumsum(axis=0).cumsum(axis=1)),0,3,3,1)
    Traceback (most recent call last):
    ...
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")
    Exception: ERROR, wrong parameters: m >= n or i >= j 
    '''

    # checking needed parameters for running function
    if m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")
    # print("cumsum\n", cum_a)
    # Inclusion-Exclusion principle for n = 2 dimensions
    so = cum_a[n, j]
    s1 = -cum_a[n, i]
    s2 = -cum_a[m, j]
    s3 = cum_a[m, i]

    return so + s1 + s2 + s3

def simple_summed_area_table(array,m,n,i,j):
    '''
    calculate summed_area_table using standart np.sum() function
    examples:
    >>> simple_summed_area_table(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]),0,4,0,5)
    606
    >>> simple_summed_area_table(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]),1,4,2,5)
    478
    >>> simple_summed_area_table(np.array([[1,3,6,10,12],[4,9,17,27,32],[11,24,41,61,70],[15,33,53,84,93]]),0,3,3,1)
    Traceback (most recent call last):
    ...
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")
    Exception: ERROR, wrong parameters: m >= n or i >= j 
    '''
    if m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: m >= n or i >= j ")
    # made needed slice from 2d-array and summed it 
    res = np.sum(array[m:n,i:j])
    return res



def summed_volume_table(cum_a, l, k, m, n, i, j):
    '''
    returns summed volume table of array[l:k,m:n,i:j]
    function receives seven  parameters: cumulatived sum-array, (x0, y0 ,z0, x,y,z) - limits of 3D figure
    calculated the Inclusion-Exclusion principle

    examples:
    >>> summed_volume_table(adding_zeros(np.array([[[1,3,6,10],[3,8,14,21],[8,19,28,40]],[[8,18,30,39],[14,34,53,70],[22,49,71,94]],[[11,28,49,66],[18,49,79,109],[27,72,109,151]]]).cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)),0,3,0,3,1,4)
    1319
    >>> summed_volume_table(adding_zeros(np.array([[[1,3,6,10],[3,8,14,21],[8,19,28,40]],[[8,18,30,39],[14,34,53,70],[22,49,71,94]],[[11,28,49,66],[18,49,79,109],[27,72,109,151]]]).cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)))
    Traceback (most recent call last):
    ...
    TypeError: summed_volume_table() missing 6 required positional arguments: 'l', 'k', 'm', 'n', 'i', and 'j'
    '''

    if l >= k or m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: l >=k or m >= n or i >= j ")
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
    
    return s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7


def simple_summed_volume_table(array,l,k,m,n,i,j):
    '''
    calculate summed_volume_table using standart np.sum() function
    examples:
    >>> simple_summed_volume_table(np.array([[[1,3,6,10],[3,8,14,21],[8,19,28,40]],[[8,18,30,39],[14,34,53,70],[22,49,71,94]],[[11,28,49,66],[18,49,79,109],[27,72,109,151]]]),0,3,1,4,1,4)
    1070
    >>> simple_summed_volume_table(np.array([[[1,3,6,10],[3,8,14,21],[8,19,28,40]],[[8,18,30,39],[14,34,53,70],[22,49,71,94]],[[11,28,49,66],[18,49,79,109],[27,72,109,151]]]))
    Traceback (most recent call last):
    ...
    TypeError: simple_summed_volume_table() missing 6 required positional arguments: 'l', 'k', 'm', 'n', 'i', and 'j'
    '''
    if l >= k or m >= n or i >= j:
        raise Exception("ERROR, wrong parameters: l >=k or m >= n or i >= j ")
    # made needed slice from 3d array and summed it 
    res = np.sum(array[l:k,m:n,i:j])
    return res

def adding_zeros(cum_a):
    """
    returns cum_a with added zero rows and columns in order to exclude special cases( for example: m == 0, n == 0, and so on) 
    """
    new_shape = tuple(np.array(cum_a.shape) + np.array( [1]*len(cum_a.shape) ))
    zeros = np.zeros(new_shape, dtype="int64")
    d = 'zeros{}'.format("[" +",".join(  ['1:']*len(cum_a.shape)  ) + "]")
    exec(d +'=cum_a')
    return zeros

example_1 = np.random.randint(1, 50, (100))
cum_example_1 = adding_zeros(example_1.cumsum(axis=0))

example_2 = np.random.randint(1, 50, (7,7))
cum_example_2 = adding_zeros(example_2.cumsum(axis=0).cumsum(axis=1))

example_3 = np.random.randint(1, 50, (5, 7, 8))
cum_example_3 = adding_zeros(example_3.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2))


one_dim = partial_sum(cum_example_1, 2, 50)
print(one_dim)

two_dim = summed_area_table(cum_example_2, 0, 5, 1, 3)
print(two_dim)

three_dim = summed_volume_table(cum_example_3,0,5,0,5,1,4)
print(three_dim)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
