#for multiple neurons
inputsMul = [1, 2, 3, 4]
weights1 = [0.3, 0.4, 0.9, -0.5]
weights2 = [0.4, 0.6, -0.8, 0.2]
weights3 = [0.7, -0.1, 0.9, 0.2]

bias1 = 2.0
bias2 = 3.0
bias3 = 0.5

outputMul = [inputsMul[0]*weights1[0] + inputsMul[1]*weights1[1] + inputsMul[2]*weights1[2] + inputsMul[3]*weights1[3] + bias1, 
          inputsMul[0]*weights2[0] + inputsMul[1]*weights2[1] + inputsMul[2]*weights2[2] + inputsMul[3]*weights2[3] + bias2,
          inputsMul[0]*weights3[0] + inputsMul[1]*weights3[1] + inputsMul[2]*weights3[2] + inputsMul[3]*weights3[3] + bias3]

print(outputMul)