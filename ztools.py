import cupy as cp
import matplotlib.pyplot as plt

## ====================================================================================
#   ZTools Layers
## ====================================================================================

# - Layer Dense -----------------------------------------------------------------------
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons,
                weight_regularizer_l1=0., weight_regularizer_l2=0., 
                bias_regularizer_l1=0., bias_regularizer_l2=0.):

        # Initialize weights and biases
        self.weights = 0.01 * cp.random.randn(n_inputs, n_neurons)
        self.biases  = cp.zeros((1, n_neurons))
        self.dinputs = 0
        
        # Set regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1   = bias_regularizer_l1
        self.bias_regularizer_l2   = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        self.output = cp.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradient on parameters
        self.dweights = cp.dot(self.inputs.T, dvalues)
        self.dbiases  = cp.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on regularization
        if self.weight_regularizer_l1 > 0:
            dL1 = cp.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        if self.bias_regularizer_l1 > 0:
            dL1 = cp.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # Gradient on values
        self.dinputs  = cp.dot(dvalues, self.weights.T)

# - Layer_Input	- Dummy layer to ease the implementation-------------------------------
class Layer_Input:
    # Forward pass
    def forward(self, inputs):
        self.output = inputs

# - Activation_ReLU -------------------------------------------------------------------
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs

        # Calculate output values from inputs
        self.output = cp.maximum(0, inputs)
        self.dinputs = 0

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable, 
        # let's make a copy of the variable first 
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for output
    def predictions(self, outputs):
        return outputs

# - Activation_Softmax ----------------------------------------------------------------
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1,1)
            # Calculate Jacobian matrix of the output ...
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    # Calculate predictions for output
    def predictions(self, outputs):
        return cp.argmax(outputs, axis=1)

# - Activation_Sigmoid ----------------------------------------------------------------
class Activation_Sigmoid: 

    # Calculate predictions for output
    def predictions(self, outputs):
        return (outputs > 0.5) * 1     # '*1' converts 'True/False' to '1/0'

# - Activation_Linear -----------------------------------------------------------------
class Activation_Linear:

    # Forward pass
    def forward(self, inputs): 
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # Derivative is 1, 1*dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for output
    def predictions(self, outputs):
        return outputs

# - Activation_Softmax_Loss_Categoricalentropy ----------------------------------------

# - Optimizer_Adam --------------------------------------------------------------------
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
            layer.weight_momentums = cp.zeros_like(layer.weights)
            layer.weight_cache     = cp.zeros_like(layer.weights)
            layer.bias_momentums   = cp.zeros_like(layer.biases)
            layer.bias_cache       = cp.zeros_like(layer.biases)

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

# - Optimizer_SGD ---------------------------------------------------------------------

# - class Optimizer_Adagrad -----------------------------------------------------------

# - class Optimizer_RMSprop -----------------------------------------------------------

# - Loss-------------------------------------------------------------------------------
class Loss:

    # Regularization loss calculation
    def regularization_loss(self):
        
        regularization_loss = 0        
        
        # Iterate all trainable layers
        for layer in self.trainable_layers:
            
            # L1/L2 regularization (if factor greater than 0)
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * cp.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * cp.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * cp.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * cp.sum(layer.biases * layer.biases)

            return regularization_loss
    
    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = cp.mean(sample_losses)

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    def forward(self, y_pred, y_true):
        return []

# - Loss_CategoricalCrossentropy ------------------------------------------------------

# - Loss_MeanSquaredError -------------------------------------------------------------
class Loss_MeanSquaredError(Loss):   # L2 Loss
    
    # Forward pass
    def forward(self, y_pred, y_true): 
        # Calculate loss
        sample_losses = cp.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
 
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples

# - Loss_MeanAbsoluteError ------------------------------------------------------------
class Loss_MeanAbsoluteError(Loss):   # L1 loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = cp.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):

        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = cp.sign(y_true - dvalues) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
# - Accuracy --------------------------------------------------------------------------
class Accuracy:

    # Calculate an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = cp.mean(comparisons)

        return accuracy

# - Accuracy_Regression ---------------------------------------------------------------
class Accuracy_Regression(Accuracy):

    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculates precision value based on passed-in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = cp.std(y) / 250

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return cp.absolute(predictions - y) < self.precision
    
# - Model -----------------------------------------------------------------------------
class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss      = loss
        self.optimizer = optimizer
        self.accuracy  = accuracy

    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects 
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If first layer, then use 'Layer_Input' as first layer
            if i==0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count -1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)
 
    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1,epochs+1):
        
            # Forward pass
            output = self.forward(X)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy    = self.accuracy.calculate(predictions, y)

            # Perform backward pass
            self. backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch {epoch:<5d} | ' +
                      f'acc: {accuracy:.3f} |' +
                      f'loss: {loss:.3f} [' +
                      f'data_loss: {data_loss:.3f} |' +
                      f'reg_loss: {regularization_loss:.3f} ] |' +
                      f'lr: {self.optimizer.current_learning_rate:.5f} ') 

    # Forward pass
    def forward(self, X):
        # Mine is faster, of course .... and better. Just saying :) 
        self.input_layer.forward(X)    # Our dummy layer, fill the output parameter

        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        return layer.output

    # Backward pass
    def backward(self, output, y):

        # First call backward method on the loss this will set 
        # dinputs property that the last layer will try to 
        # access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects in 
        # reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

# - plt_multiclass_decision_boundary --------------------------------------------------
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
    #-# dense1.forward(X_test)
    #-# activation1.forward(dense1.output)
    #-# dense2.forward(activation1.output)

    # Annoyingly I need to convert to PyTorch Tensor and use F.softmax <-- ToDo!
    y_pred = F.softmax(torch.from_numpy(dense2.output), dim=1)
    _, y_pred = y_pred.max(dim=1)
    y_pred = y_pred.reshape(xx.shape)
    y_pred = y_pred.numpy()
    
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

