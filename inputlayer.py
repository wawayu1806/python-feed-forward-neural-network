from neuron import Neuron
import numpy as np

class InputLayer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.neurons = [Neuron(0.0) for _ in range(self.n_neurons)] 
        self.input_vector = None
        self.weight_matrix = None 

    def initialize_weight_matrix(self, next_layer_n):
        mean = 0
        stddev = np.sqrt(2/(self.n_neurons+next_layer_n))
        mat = np.random.normal(loc=mean, scale=stddev, size=(next_layer_n, self.n_neurons))
        self.weight_matrix = mat 

    def set_input_vector(self, z):
        z.shape = (self.n_neurons, 1)
        self.input_vector = z

    def get_input_vector(self):
        return self.input_vector

    def get_weight_matrix(self):
        return self.weight_matrix

    def update_weights(self, weights):
        self.weight_matrix = weights