import numpy as np

#batches of input

inputsBatch = [[1, 2, 3, 4],
               [2, 3, 4, 5],
               [5, 6, 7, 8]]
weightsSim = [[0.3, 0.4, 0.9, -0.5], 
              [0.4, 0.6, -0.8, 0.2], 
              [0.7, -0.1, 0.9, 0.2]]

biases = [2.0, 3.0, 0.5]

numpyBatchOutput = np.dot(inputsBatch, np.array(weightsSim).T) + biases
print(numpyBatchOutput)

#here, if we do not take the transpose of weightsSim, we will get the error - 
# ValueError: shapes (3,4) and (3,4) not aligned: 4 (dim 1) != 3 (dim 0)
#because the number of rows of the first matrix is not equal to the number of the columns of the 2nd matrix

#Adding layers:

weightsSim2 = [[0.1, -0.14, 0.5], 
               [-0.5, 0.12, -0.33], 
               [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_numpyBatchOutput = np.dot(inputsBatch, np.array(weightsSim).T) + biases
layer2_numpyBatchOutput = np.dot(layer1_numpyBatchOutput, np.array(weightsSim).T) + biases

#so here, we are adding a layer and for the second layer, the input layer will be the output of the 1st layer.
#but this is too shabby, so we will create an object for this