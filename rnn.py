import numpy as np

#Activation function
def ReLU(x):
    return np.maximum(x, 0)

def ReLU_deriv(x):
    return x > 0

#inputs according to assumptions
X = np.array([
    [[0.0], [0.0]], #low, low -> low
    [[0.0], [1,0]], #low, high -> higher
    [[1.0], [0.4]], #high, low -> lower
    [[1.0], [1.0]]  #high, high -> high
], dtype=float)

Y = np.array([
    [0.0], 
    [1.2],
    [0.2],
    [1.0]
], dtype=float)

#defining parameters
np.random.seed(1)
input_size = 1
hidden_size = 1
ouput_size = 1
lr = 0.01 # learning rate
epochs = 2000
grad_clip = 5.0

def init_params():
    w1 = np.random.randn(hidden_size, input_size) * 0.1
    w2 = np.random.randn(hidden_size, input_size) * 0.1
    w3 = np.random.randn(hidden_size, input_size) * 0.1
    b1 = np.zeros((hidden_size, 1))
    b2 = np.zeros((hidden_size, 1))

    return w1, w2, w3, b1, b2

w1, w2, w3, w4, b1, b2 = init_params()

#forward pass (input * weights + bias)
def forward(x_seq):
    T = x_seq.shape[0] # x_seq is the input (0, 1, etc) -> (today, tomorrow, yesterday, etc)
    h_prev = np.zeros((hidden_size, 1)) # a1, a2, a3, etc
    hs = [] #hidden states h_t
    zs = [] #pre-activation z_t

    for i in range(T):
        x_t = x_seq[i].reshape(-1, 1)
        z_t = np.dot(w1, x_t) + np.dot(w2, h_prev) + b1
        h_t = ReLU(z_t)
        zs.append(z_t)
        hs.append(h_t)
        h_prev = h_t

    y_pred = np.dot(w3, h_prev) + b2

    return y_pred, hs, zs

'''
TRACING - 
let w1 = 1.5, w2 = 0.5, b1 = 0, b2 = 0, w3 = 1.8
x_seq = [[0.0], [0.0]]
shape of x_seq = (2, 1)
T = x_seq.shape[0] 
T = 2
h_prev = np.zeros((hidden_size, 1))
hidden_size = 8
h_prev = [[0]]

hs = []
zs = []

for i in range(T):
T = 2
    -> i = 0
    x_t = x_seq[i].reshape(-1, 1)
    x_t = x_seq[0].reshape(-1, 1)
    x_t = [[0.0]]

    how?
    x_seq[0] = [0.0], and its shape is (1,) which means (1 row, 0 columns)
    reshape(-1, 1) changes it to a column vector of shape (1, 1)
    this is done so that the matrix multiplication can be done. 

    z_t = np.dot(w1, x_t) + np.dot(w2, h_prev) + b1
    z_t = np.dot(1.5, [[0.0]]) + np.dot(0.5, [[0]]) + 0

    for simplicity, we can take these as scalars and not vectors. 

    multiplying, we get ->
    z_t = 0 + 0 + 0
    z_t = 0

    h_t = ReLU(z_t)
    h_t = 0

    zs.append(z_t)
    zs = [0]

    hs = append(h_t)
    hs = [0]

    h_prev = h_t
    h_prev = 0

    -> i = 1
    x_t = x_seq[i].reshape(-1, 1)
    x_t = x_seq[1].reshape(-1, 1)
    x_t = [[0.0]]

    z_t = np.dot(w1, x_t) + np.dot(w2, h_prev) + b1
    z_t = np.dot(1.5, [[0.0]]) + np.dot(0.5, [[0]]) + 0

    multiplying, we get ->
    z_t = 0 + 0 + 0
    z_t = 0

    h_t = ReLU(z_t)
    h_t = 0

    zs.append(z_t)
    zs = [0, 0]

    hs = append(h_t)
    hs = [0, 0]

    h_prev = h_t
    h_prev = 0

exit loop

y_pred = np.dot(w3, h_prev) + b2
y_pred = np.dot(1.8, 0) + 0
y_pred = 0 + 0
y_pred = 0

'''


