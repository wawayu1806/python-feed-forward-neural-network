import numpy as np
from neuron import Neuron 

class Layer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.neurons = [Neuron(0.0) for _ in range(self.n_neurons)] 
        self.bias_vector = np.zeros(shape=(self.n_neurons,1))
        self.input_vector = None
        self.weight_matrix = None 

    def initialize_weight_matrix(self, m):
        mean = 0
        stddev = np.sqrt(2/(self.n_neurons+m))
        mat = np.random.normal(loc=mean, scale=stddev, size=(self.n_neurons, m))
        self.weight_matrix = mat 

    def set_bias_vector(self):
        for i, neuron in enumerate(self.neurons):
            self.bias_vector[i,0] = neuron.get_bias() 

    def set_input_vector(self, z):
        self.input_vector = z

    def get_bias_vector(self):
        return self.bias_vector

    def update_biases(self, biases):
        for neuron, bias in zip(self.neurons, biases):
            neuron.set_bias(bias)

    def update_weights(self, weights):
        self.weight_matrix = weights

    def forward(self):
        z = np.dot(self.weight_matrix, self.input_vector) + self.bias_vector
        activation_output =  Neuron.sigmoid(z)
        return activation_output
