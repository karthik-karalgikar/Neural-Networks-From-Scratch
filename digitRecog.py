import struct
import numpy as np
import csv
import pandas as pd
from matplotlib import pyplot as plt

'''
CONVERTING IDX FILES TO CSV (IDC FILES ARE BINARY FILES IN IDX FORMAT)
idc file format is a simple binary format used to store vectors and multidimensional matrices of various numberical types

def load_images(images_path):
    with open(images_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file: {images_path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows * cols)  # flatten each 28x28 → 784
    return data

def load_labels(labels_path):
    with open(labels_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file: {labels_path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def convert_to_csv(images_path, labels_path, output_path):
    images = load_images(images_path)
    labels = load_labels(labels_path)

    print(f"Converting {len(labels)} samples to {output_path}...")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write each row: label + 784 pixel values
        for i in range(len(labels)):
            row = [labels[i]] + images[i].tolist()
            writer.writerow(row)

    print(f"Done! Saved to {output_path}")

# Paths to your files
train_images = "input/train-images-idx3-ubyte/train-images-idx3-ubyte"
train_labels = "input/train-labels-idx1-ubyte/train-labels-idx1-ubyte"
test_images  = "input/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
test_labels  = "input/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"

# Convert both train and test sets
convert_to_csv(train_images, train_labels, "mnist_train.csv")
convert_to_csv(test_images, test_labels, "mnist_test.csv")

'''

data = pd.read_csv("mnist_train.csv", header = None)
#loads the CSV into a pandas dataframe
#header = None -> because the CSV has no header row

data = np.array(data)
#Converts dataframe into numpy array

m, n = data.shape
#m = 60000, n = 785 (1 label column + 784 features)

np.random.shuffle(data)
#Shuffles the dataset in-place so training and validation data are mixed randomly

data_cross_val = data[0:1000].T
Y_val = data_cross_val[0]
X_val = data_cross_val[1:n]
X_val = X_val / 255.
'''
Splitting the validation set:
data[0:1000] takes the first 1000 rows as validation set
.T -> takes the transpose, so shape changes from (1000, 785) to (785, 1000)
Each column is a sample.
Y_val = data_cross_val[0] -> first row(after transpose) = labels. Shape - (1000, )
X_val = data_cross_val[1:n] -> the remaining 784 rows = pixel values. Shape - (784, 1000)
'''

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
'''
Splitting the training set:
data[1000:m] takes the remaining 59000 rows as training set
.T -> takes the transpose, so shape changes from (59000, 785) to (785, 59000)
Each columnn is a sample
Y_train = data_train[0] -> first row = labels, Shape - (1000, )
X_train = data_train[1:n] -> the remaining 784 rows = pixel values. Shape - (784, 59000)
'''

#print(Y_train)
#[8 3 1 ... 3 7 6]

#print(X_train[0])
#[0 0 0 ... 0 0 0]

#print(X_train[0].shape)
#(59000,)

#print(X_train[:, 0].shape)
#(784,)

def init_params():
    W1 = np.random.rand(10, 784) * np.sqrt(1/784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10) * np.sqrt(1/10)
    b2 = np.random.rand(10, 1)

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis = 0))
    return expZ / expZ.sum(axis = 0, keepdims = True)

def forward_prop(W1 ,b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T

    return one_hot_Y

def ReLU_deriv(Z):
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)

    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration:", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))

    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
