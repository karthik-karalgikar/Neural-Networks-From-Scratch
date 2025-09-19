import numpy as np

inputs = [1, 2, 3, 4]
weights = [0.2, 0.3, 0.5, 0.8]
bias = 2.0

numpyOutput = np.dot(inputs, weights) + bias

print(numpyOutput)

#numpyLayers

inputsMul = [1, 2, 3, 4]
weightsSim = [[0.3, 0.4, 0.9, -0.5], 
              [0.4, 0.6, -0.8, 0.2], 
              [0.7, -0.1, 0.9, 0.2]]

biases = [2.0, 3.0, 0.5]

numpyLayerOutput = np.dot(weightsSim, inputsMul) + biases
print(numpyLayerOutput)

#refer pg 26 of the book for visualisation