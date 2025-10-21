#Pg 63

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X = [[1, 2, 3, 4],
    [2, 3, 4, 5],
    [5, 6, 7, 8]]

# Create dataset
X, y = spiral_data(samples=100, classes=3)

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        #initialize inputs and weights
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    '''
    np.random.randn produces a Gaussian distribution with a mean of 0 and a variance of 1, 
    which means that it'll generate random numbers, positive and negative, centered at 0 and with the mean value close to 0
    We're going to multiply this Gaussian distribution for the weights by 0.01 
    to generate numbers that are a couple of magnitudes smaller. 
    Otherwise, the model will take more time to fit the data during the training process as starting values 
    will be disproportionately large compared to the updates being made during training.

    so if I do "print(0.10 * np.random.randn(4,3))", then this means that there are 4 inputs for 3 neurons. 
    (check the multipleNeurons image for visualization)
 
    '''

    def forward(self, inputs):
        #Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

# layer2 = Layer_Dense(5, 3)

layer1.forward(X)

print(layer1.output)

activation1.forward(layer1.output)
print(activation1.output)
# layer2.forward(layer1.output)

# print(layer2.output)

# print(layer1.weights)
# print(layer1.biases)
# print(layer1.output)

'''
[[-0.0103796  -0.00032591 -0.00642841 -0.00348842 -0.01115847]
 [-0.00865783 -0.00653263 -0.00097961  0.02552741  0.00959843]
 [ 0.00964404 -0.00619626  0.00612478  0.00114433  0.00722092]
 [-0.00347226 -0.01102576 -0.0035272  -0.00159562 -0.00077808]]

[[0. 0. 0. 0. 0.]]

[[-0.01265217 -0.07608299 -0.00412209  0.04461695  0.02658884]
 [-0.02551782 -0.10016356 -0.00893254  0.06620466  0.03147164]
 [-0.06411476 -0.17240525 -0.02336386  0.13096781  0.04612005]]
'''


# # Create Dense layer with 2 input features and 3 output values
# dense1 = Layer_Dense(2, 3)

# # Perform a forward pass of our training data through this layer
# dense1.forward(X)

# # output of the first few samples:
# print(dense1.output[:5])

'''
[[ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
 [-1.0475188e-04  1.1395361e-04 -4.7983500e-05]
 [-2.7414842e-04  3.1729150e-04 -8.6921798e-05]
 [-4.2188365e-04  5.2666257e-04 -5.5912682e-05]
 [-5.7707680e-04  7.1401405e-04 -8.9430439e-05]]
'''