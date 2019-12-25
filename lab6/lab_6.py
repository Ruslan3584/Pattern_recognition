from functions import *

a,b,r = parameters()
data = get_data(50)
s_data = split_data(data,a,b,r)

result = perceptron(s_data)
print('result_parameters',result)
print('original_parameters',a,b,r)
