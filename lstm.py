import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return(np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class LSTM:
    def __init__(self, hidden_size, input_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_f = np.random.randn(hidden_size, input_size)
        self.w_i = np.random.randn(hidden_size, input_size)
        self.w_c = np.random.randn(hidden_size, input_size)
        self.w_o = np.random.randn(hidden_size, input_size)
       
        self.u_f = np.random.randn(hidden_size, hidden_size)
        self.u_i = np.random.randn(hidden_size, hidden_size)
        self.u_c = np.random.randn(hidden_size, hidden_size)
        self.u_o = np.random.randn(hidden_size, hidden_size)

        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))


    def forward(self, x_seq):
        T = x_seq.shape[0]
        h_t = np.zeros((self.hidden_size, 1))
        c_t = np.zeros((self.hidden_size, 1))

        h_list = []
        c_list = []

        for x_t in x_seq:
            x_t = x_t.reshape(-1, 1)

            #forget gate
            f_t = sigmoid(np.dot(self.w_f, x_t) + np.dot(self.u_f, h_t) + self.b_f)

            #input gate
            i_t = sigmoid(np.dot(self.w_i, x_t) + np.dot(self.u_i, h_t) + self.b_i)

            #candidate memory
            candidate = tanh(np.dot(self.w_c, x_t) + np.dot(self.u_c, h_t) + self.b_c)

            # long term memory
            c_t = (i_t * candidate) + (f_t * c_t)

            #output gate
            o_t = sigmoid(np.dot(self.w_o, x_t) + np.dot(self.u_o, h_t) + self.b_o)

            h_t = o_t * tanh(c_t)

            h_list.append(h_t.copy())
            c_list.append(c_t.copy())

        return h_t, h_list, c_list

lstm = LSTM(hidden_size=3, input_size=1)
x_seq = np.array([[1.0], [0.5], [0.25], [1.0]])

h_last, h_list, c_list = lstm.forward(x_seq)

# Output layer for prediction
W_y = np.random.randn(1, 3)
b_y = np.zeros((1, 1))

# Predict Day 5
y_pred = np.dot(W_y, h_last) + b_y

print(y_pred[0][0])

print("\nHidden states (short-term memory):")
for i, h in enumerate(h_list):
    print(f"Day {i+1}: {h.ravel()}")

print("\nCell states (long-term memory):")
for i, c in enumerate(c_list):
    print(f"Day {i+1}: {c.ravel()}")


'''
Output = 

0.9847199921487875

Hidden states (short-term memory):
Day 1: [ 0.06162929  0.46189961 -0.09151756]
Day 2: [ 0.03821676  0.34025179 -0.01620486]
Day 3: [0.03559226 0.29765014 0.01335178]
Day 4: [ 0.075589    0.50936482 -0.04602351]

Cell states (long-term memory):
Day 1: [ 0.19506637  0.74096967 -0.27138356]
Day 2: [ 0.1260089   0.88596291 -0.04317269]
Day 3: [0.09586609 0.78194634 0.03135481]
Day 4: [ 0.30024285  1.04858921 -0.14551363]
'''

#You might get a different output and I have not implemented backpropagation to tweak the weights and biases, so 
#there is a very high chance we will get the wrong answer. 