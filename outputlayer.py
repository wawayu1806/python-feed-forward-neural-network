from neuron import Neuron
import numpy as np

class OutputLayer:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.neurons = [Neuron(0.0) for _ in range(self.n_neurons)] 
        self.bias_vector = np.zeros(shape=(self.n_neurons,1))
        self.input_vector = None

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

