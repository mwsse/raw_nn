## ===================================================================================
#  raw-nn
#  From the course: Neural Networks from scratch in Python 2020
#  (Page 238+ for copy)
## ==================================================================================== 

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from timeit import timeit

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from torch import nn
import torch.nn.functional as F

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

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs           # Remember input values

        # Get unnormalized probabilities 
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create unitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1,1)
            # Calculate Jacobian matrix of the outpt and ...
            jacobian_matrix = np.diagflat(single_output)-np.dot(single_output, single_output.T)

            # ... calculate sample-wise gradient and add it to the
            # array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# - Common loss class
class Loss:

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        return np.mean(sample_losses)

    def forward(self, y_pred, y_true):
        print("Error in 'class Loss'. you must override forward()")
        return []

# - Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0. Clip both sides to not
        # drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample.
        # First sample is used to count them
        labels = len(dvalues[0])

        # If labels are spare, turn them into one-hot vector
        # E.g. [(0,0,1,2)] -->  [(1,0,0),(1,0,0),(0,1,0),(0,0,1)]
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# - Softmax classifier
#   Combined Softmax activiation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects 
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of sampels
        samples = len(dvalues)

        # If labels are one-hot encoded, turm then into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify variable
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# - Optimizer SGD - Stochastic Gradient Descent
class Optimizer_SGD:

    # Initialize optimizer - set settings, learning rate of 1 is default
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate          = learning_rate
        self.current_learning_rate  = learning_rate
        self.decay                  = decay
        self.iterations             = 0
        self.momentum               = momentum

    # Call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        if self.momentum:

            # Create momentum arrays if layer does not have them
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases )

            # Build weight updates with momentum.
            # Take previous updates * retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Else run vanilla SGD update
        else:                 
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates   = -self.current_learning_rate * layer.dbiases


        layer.weights += weight_updates
        layer.biases  += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# - Optimizer AdaGrad - [Adaptive Gradient]
class Optimizer_Adagrad:
    
    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate          = learning_rate
        self.current_learning_rate  = learning_rate
        self.decay                  = decay
        self.iterations             = 0
        self.epsilon                = epsilon

    # Call once before any parameters updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # Add cache array if layer does not have them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases )

        # Update cache with squared current gradient
        layer.weight_cache += layer.dweights**2
        layer.bias_cache   += layer.dbiases**2

        # Vanilla SGD Parameter update + normalization with square root cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  += -self.current_learning_rate * layer.dbiases  / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# - Optimizer RMSprop - [Root Mean Square Propagation]
class Optimizer_RMSprop:

    # Initialize optimizer
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay*self.iterations))
    
    # Update params
    def update_params(self, layer):

        # Add cache array if layer does not have them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache   = np.zeros_like(layer.biases )

        # Update cache with squared current gradient
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache   = self.rho * layer.bias_cache   + (1-self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization with squared cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases  += -self.current_learning_rate * layer.dbiases  / (np.sqrt(layer.bias_cache)   + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# - Optimizer Adam - [Adaptive Momentum]
class Optimizer_Adam:

    # Initialize optimizer
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate          = learning_rate
        self.current_learning_rate  = learning_rate
        self.decay                  = decay
        self.iterations             = 0
        self.epsilon                = epsilon
        self.beta_1                 = beta_1
        self.beta_2                 = beta_2

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = \
                self.learning_rate * (1.0 / (1.0+self.decay * self.iterations))
    
    # Update parameters
    def update_params(self, layer):

        # Create cache arrays if not exist
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.bias_cache       = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = \
            self.beta_1 * layer.weight_momentums + (1-self.beta_1)*layer.dweights
        layer.bias_momentums = \
            self.beta_1 * layer.bias_momentums   + (1-self.beta_1)*layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations+1))
        bias_momentums_corrected   = layer.bias_momentums   / (1 - self.beta_1 ** (self.iterations+1))

        # Update cache with squared current gradients
        layer.weight_cache = \
            self.beta_2 * layer.weight_cache + (1-self.beta_2)*layer.dweights**2
        layer.bias_cache = \
            self.beta_2 * layer.bias_cache   + (1-self.beta_2)*layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1-self.beta_2**(self.iterations+1))
        bias_cache_corrected   = layer.bias_cache   / (1-self.beta_2**(self.iterations+1))

        # Vanilla SGD parameter update + normalization with sqrt cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases  += -self.current_learning_rate * bias_momentums_corrected   / (np.sqrt(bias_cache_corrected)   + self.epsilon)

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

## ----------------------------------------------------------------------- 
#   plot_multiclass_decision_boundary(model, X, y) 
## -----------------------------------------------------------------------
def plot_multiclass_decision_boundary(X, y):
    """
    source: https://madewithml.com/courses/foundations/neural-networks/
    """
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    t = np.linspace(x_min, x_max, 101)
    xx,yy = np.meshgrid(np.linspace(y_min, y_max, 101), np.linspace(y_min, y_max, 101))
    cmap = plt.cm.Spectral

    X_test = np.c_[xx.ravel(), yy.ravel()]
    # Running the model should be it's own function
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    # Annoyingly I need to convert to PyTorch Tensor and use F.softmax <-- ToDo!
    y_pred = F.softmax(torch.from_numpy(dense2.output), dim=1)
    _, y_pred = y_pred.max(dim=1)
    y_pred = y_pred.reshape(xx.shape)
    y_pred = y_pred.numpy()
    
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

## ------------------------------------------------------------------------------------
#   Main part - Training, testing, evaluating
## ------------------------------------------------------------------------------------

# Create dataset using 'spiral_data' from nnfs
X, y = spiral_data(samples=100, classes=3)

# Create network and define loss, accuracy and optimization functions (1x64)
dense1          = Layer_Dense(2,64)
activation1     = Activation_RelU()
dense2          = Layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
#optimizer      = Optimizer_SGD(decay=1e-3, momentum=0.9)
#optimizer      = Optimizer_Adagrad(decay=1e-4)
#optimizer      = Optimizer_RMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer       = Optimizer_Adam(learning_rate=0.02, decay=1e-5)

# Start the training
for epoch in range(10001):

    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy from output of loss_activation and targets
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2: 
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 1000:
        print(f"epoch: {epoch:5d} | acc: {accuracy:.3f} | ", end="")
        print(f"loss: {loss:.3f} | lr: {optimizer.current_learning_rate}")

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Optimizer
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


## -------------------------------------------------------------------------------------
#   Evaluate model 
## -------------------------------------------------------------------------------------

# Visualize the decision boundary
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_multiclass_decision_boundary(X=X, y=y)
# We don't have any test or validation set yet

plt.show()
