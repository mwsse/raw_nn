## ===================================================================================
#  raw-nn
#  From the course: Neural Networks from scratch in Python 2020
#  (Page 238+ for copy)
## ==================================================================================== 

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from timeit import timeit

nnfs.init()

## ------------------------------------------------------------------------------------
#  Create the Dense layer, ReLU activation, Loss and Softmax + Optimization 
## ------------------------------------------------------------------------------------

# - Create Dense Layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs           # Remember input values
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradient on parameters and values
        self.dweights  = np.dot(self.inputs.T, dvalues)
        self.dbiases   = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs   = np.dot(dvalues, self.weights.T)

# - Create ReLU activation
class Activation_RelU:

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs           # Remember input values
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()  # Save the original variable
        self.dinputs[self.inputs <= 0] = 0

# - Softmax activation
class Activation_Softmax:

    