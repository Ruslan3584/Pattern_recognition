import numpy as np

def parameters():
    '''
    generate circle parameters
    returns: 
            a,b - center of the circle 
            r - radius

    examples:
    >>> np.random.seed(2)
    >>> parameters()
    (-5, -12, 8)
    >>> np.random.seed(135)
    >>> parameters()
    (11, 7, 8)
    '''
    a = np.random.randint(-20,20)
    b = np.random.randint(-20,20)
    r = np.random.randint(5,10)
    return a,b,r


def get_data(n):
    '''
    generate n points
    args: n - number of points(int)
    returns: array of 2D points 

    examples:
    >>> get_data(1)
    Traceback (most recent call last):
    ...
    Exception: n is too small
    >>> np.random.seed(135)
    >>> get_data(2)
    array([[  1, 170,  11,   7],
           [  1, 306,  -9, -15]])
    >>> np.random.seed(2)
    >>> get_data(2)
    array([[  1, 169,  -5, -12],
           [  1,   8,   2,  -2]])
    '''
    if n < 2:
        raise Exception('n is too small')
    data = np.array([[np.random.randint(-20,20),np.random.randint(-20,20)] for i in range(n)])
    # add ones column 
    data= np.pad(data, (2,0), 'constant', constant_values=1)
    data = data[2:]
    data[:,1] = np.array([ data[i][2]**2 + data[i][3]**2 for i in range(len(data))])
    return data



def split_data(data,a,b,r):
    '''
    splits the data in two parts: inside and outside the circle
    args: data  - array of 2D points
          a,b,r - original parameters of the circle
    returns: 2D array (points inside and outside the circle),points inside the circle are reversed

    examples:
    >>> split_data(np.array([[1,169,-5,-12],[1,8,2,  -2]]),0,0,0)
    Traceback (most recent call last):
    ...
    Exception: radius less or equal to zero
    '''
    if r <= 0:
        raise Exception('radius less or equal to zero')
    in_circle = []
    out_circle = []
    for i in range(len(data)):
        if (data[i][2] - a)**2 + (data[i][3] - b)**2 < r**2 :
            in_circle.append(data[i])

        elif (data[i][2] - a)**2 + (data[i][3] - b)**2 > r**2  :
            out_circle.append(data[i])
    if len(in_circle) == 0:
        raise Exception('there is no points inside the circle')
    # reverse inside_points
    in_circle = -np.array(in_circle)
    # add additional point to avoid 'divide by zero' problem
    in_circle = np.vstack((in_circle,np.array([0,1,0,0])))
    # combine one array 
    out_circle = np.array(out_circle)
    s = np.vstack((in_circle,out_circle))
    return s



def perceptron(s_data):
    '''
    realize perceptron algorithm
    args: s_data -splitted data(points inside and outside the circle)
    returns: parameters of the circle
    '''
    # initialize alpha
    alpha = np.zeros(4)
    while ((s_data.dot(alpha)) <= 0).any():
        for i in range(len(s_data)):
            if s_data[i].dot(alpha) <= 0:
                alpha += s_data[i]
    # normalize alpha
    alpha = alpha/alpha[1]
    # get circle parameters from alpha(result of perceptron)
    a_t = alpha[2]*(-0.5)
    b_t = alpha[3]*(-0.5)
    r_t = np.sqrt((a_t**2 + b_t**2 - alpha[0]))
    return (a_t,b_t,r_t)


if __name__ == "__main__":
    import doctest
    doctest.testmod()