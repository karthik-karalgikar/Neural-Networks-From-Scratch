#Simplifying:

inputsMul = [1, 2, 3, 4]
weightsSim = [[0.3, 0.4, 0.9, -0.5], 
              [0.4, 0.6, -0.8, 0.2], 
              [0.7, -0.1, 0.9, 0.2]]

biases = [2.0, 3.0, 0.5]

layer_outputs = []
for neuron_weights, neuron_bias in zip(weightsSim, biases):
    neuron_output = 0
    for n_input, weight in zip(inputsMul, neuron_weights):
        neuron_output = neuron_output + (n_input * weight)
    neuron_output = neuron_output + neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

'''
TRACING - 

layer_outputs = []
for neuron_weights, neuron_bias in zip(weightsSim, biases):
    neuron_weights, neuron_bias = [[0.3, 0.4, 0.9, -0.5],[2.0]] [[0.4, 0.6, -0.8, 0.2],[3.0]] [[0.7, -0.1, 0.9, 0.2], [0.5]]

    neuron_output = 0
    for n_input, weight in zip(input, neuron_weights):
        n_input, weight = [[1, 0.3], [2, 0.4], [3, 0.9], [4, -0.5]]
            neuron_output = neuron_output + (n_input * weight)
            neuron_output = 0 + (1 * 0.3)
            neuron_output = 0.3

            neuron_output = neuron_output + (n_input * weight)
            neuron_output = 0.3 + (2 * 0.4) = 0.3 + 0.8
            neuron_output = 1.1

            neuron_output = neuron_output + (n_input * weight)
            neuron_output = 1.1 + (3 * 0.9) = 1.1 + 2.7
            neuron_output = 3.8

            euron_output = neuron_output + (n_input * weight)
            neuron_output = 3.8 + (4 * -0.5) = 3.8 - 2.0
            neuron_output = 1.8

        neuron_output = neuron_output + neuron_bias 
        neuron_output = 1.8 + 2.0
        neuron_output = 3.8

        layer_outputs.append(neuron_output)

        layer_outputs = [3.8]

    neuron_weights = [0.4, 0.6, -0.8, 0.2]
    neuron_output = 0
    for n_input, weight in zip(input, neuron_weights):
        n_input, weight = [[1, 0.4], [2, 0.6], [3, -0.8], [4, 0.2]]

            neuron_output = neuron_output + (n_input * weight)
            neuron_output = 0 + (1 * 0.4) = 0 + 0.4
            neuron_output = 0.4

            neuron_output = neuron_output + (n_input * weight)
            neuron_output = 0.4 + (2 * 0.6) = 0.4 + 1.2
            neuron_output = 1.6

            neuron_output = neuron_output + (n_input * weight)
            neuron_output = 1.6 + (3 * -0.8) = 1.6 - 2.4
            neuron_output = -0.8

            neuron_output = neuron_output + (n_input * weight)
            neuron_output = 0 + (4 * 0.2) = -0.8 + 0.8
            neuron_output = 0

        neuron_output = neuron_output + neuron_bias
        neuron_output = 0 + 3.0
        neuron_output = 3.0

        layer_outputs.append(neuron_output)

        layer_outputs = [3.8, 3.0]

    neuron_weights = [0.7, -0.1, 0.9, 0.2]
    neuron_output = 0
    for n_input, weight in zip(input, neuron_weights):
        n_input, weight = [[1, 0.7], [2, -0.1], [3, 0.9], [4, 0.2]]

        neuron_output = neuron_output + (n_input * weight)
        neuron_output = 0 + (1 * 0.7)
        neuron_output = 0.7

        neuron_output = neuron_output + (n_input * weight)
        neuron_output = 0.7 + (2 * -0.1) = 0.7 - 0.2
        neuron_output = 0.5

        neuron_output = neuron_output + (n_input * weight)
        neuron_output = 0.5 + (3 * 0.9) = 0.5 + 2.7
        neuron_output = 3.2

        neuron_output = neuron_output + (n_input * weight)
        neuron_output = 3.2 + (4 * 0.2) = 3.2 + 0.8
        neuron_output = 4.0

        neuron_output = neuron_output + neuron_bias
        neuron_output = 0.5 + 4.0
        neuron_output = 4.5        

    layer_outputs.append(neuron_output)

    layer_outputs = [3.8, 3.0, 4.5]
'''
