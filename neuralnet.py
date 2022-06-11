import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def MSE(y, Y):
    return np.mean((y-Y)**2)

# def accuracy(x,Y): 
#     N_samples = x.shape[0] 



class TheNeuralNetwork(object):
    def __init__(self, network_architecture, learning_rate):
        # Initialize object using network architecture 
        # 
        self.num_layers = len(network_architecture)
        self.network_architecture = network_architecture
        
        # Initialize weights and biases       
        self.biases = [ np.random.randn(number_neurons, 1) for number_neurons in network_architecture[1:]]  
        
        self.weights = [np.random.normal(0.0, num_rows**-0.5, (num_rows, num_cols))
                        for num_cols, num_rows in zip(network_architecture[:-1], network_architecture[1:])]
        
        self.learning_rate = learning_rate
        
        # Activation function is the sigmoid function
        self.activation_function = sigmoid 
    
    def feedforward(self, input_to_network):
        """Return activations, and w*activations vector for all layers if a is an input"""
         
        activation = input_to_network
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the weighted inputs to  layers, layer by layer
        
        for b, w in zip(self.biases,self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        return activations, zs
    
    def train(self, inputs_list, targets_list):
             
        x = np.array(inputs_list, ndmin  = 2).T
        y = np.array(targets_list, ndmin = 2).T
        
        # assert x, y have the right shape 
        if x.shape[0] != self.network_architecture[0]:
            raise ValueError(f'Length of input predictors {x.shape[0]} do not match number of inputs in network architecture {self.network_architecture[0]}')
        
        if y.shape[0] != self.network_architecture[-1]:
            raise ValueError(f'''Length of targets {y.shape[0]} do not match number of number of outputs
                             in network architecture {self.network_architecture[-1]}''')
        
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # for w in self.weights:
        #     print(w.shape)
        
        # for layer, nw in enumerate(nabla_w):
        #     print('layer: ', layer)
        #     print(nw.shape)
        
        # feedforward
        activations, zs = self.feedforward(x)
        y_hat = activations[-1] #zs[-1]
        
        # backward pass
        # gradient descent for weights matrix connecting second-last to output layers 
        delta = self.cost_derivative(y_hat, y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2])
        
        # print('delta: ', delta.shape)
        # gradient descent for weights matrix in all inner layers 
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1])
        
        layer = 0
        for nb, nw in zip(nabla_b, nabla_w):
            # print('layer: ', layer)
            # print(nw.shape) 
            # print(nb.shape)
            layer +=1

        self.weights = [w - self.learning_rate*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b- self.learning_rate*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        
        return (output_activations-y)
    
    def predict(self, inputs_list):
        # TODO: assert inputs have the right shape
        inputs = np.array(inputs_list, ndmin=2).T
        activation = inputs
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the weighted inputs to  layers, layer by layer
        
        for b, w in  zip(self.biases, self.weights):
            z = np.dot(w, activation) 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        return activation.T
        
        # return z[-1]