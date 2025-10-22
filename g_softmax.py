#Softmax -
# Inputs -> exponentiate -> normalize = output
#combintation of exponentiate and normalize is softmax

import math
import numpy as np

layer_output = [4.8, 1.21, 2.385]

E = math.e

exp_values = []

for output in layer_output:
    exp_values.append(E**output)

# print(exp_values) # e^x 

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base) # e^x / sum(e^x)

# print(norm_values)
# print(sum(norm_values))


#-------------------------------------------------------#

#Numpy representation - 

exp_values_numpy = np.exp(layer_output)

norm_values_numpy = exp_values_numpy / np.sum(exp_values_numpy)

# print(exp_values_numpy)
# print(norm_values_numpy)
# print(sum(norm_values_numpy))

#-------------------------------------------------------#

# Batches - 

layer_output_batches = [[4.8, 1.21, 2.385], 
                      [8.9, -1.81, 0.2], 
                      [1.41, 1.051, 0.026]
                      ]

exp_values_batches = np.exp(layer_output_batches)

norm_values_batches = exp_values_batches / np.sum(exp_values_batches, axis = 1, keepdims = True)
'''
axis = None gives the sum of all the elements in the batch, like a scalar output. 
axis = 0 gives the column wise sum (4.8 + 8.9 + 1.41)
axis = 1 gives the sum of rows, but only like [sum sum sum]

so in order to convert in batches format, we use keepdims = True. This gives output in the form 
[[sum], 
[sum], 
[sum]
]
'''

print(norm_values_batches)

