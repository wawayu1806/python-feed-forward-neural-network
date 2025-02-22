import numpy as np
from neuron import Neuron
from hiddenlayer import HiddenLayer
from inputlayer import InputLayer
from outputlayer import OutputLayer

class Network:
    def __init__(self, n_layers=2):
        self.n_layers = n_layers 
        self.layers = [] 

    def initalize_network(self, layers):
        # add the input layer 
        inputlayer = InputLayer(layers[0])
        self.layers.append(inputlayer)

        # add the hidden layers
        for i in range(1, len(layers)-1):
            self.layers.append(HiddenLayer(layers[i]))

        # add the output layer and update self.n_layers
        outputlayer = OutputLayer(layers[-1])
        self.layers.append(outputlayer)
        self.n_layers = len(self.layers) 

    def connect_layers(self):
        for i in range(self.n_layers-1):
            self.layers[i].initialize_weight_matrix(self.layers[i+1].n_neurons)

    def __forward(self, layer_i, layer_j):
        x = layer_i.get_input_vector() 
        W = layer_i.get_weight_matrix()
        b = layer_j.get_bias_vector()

        z = np.dot(W, x) + b 
        y = Neuron.sigmoid(z)

        return y

    def feed_forward(self, input_vector):
        # set the input of the inputlayer
        self.layers[0].set_input_vector(input_vector)

        # call forward on consecutive layers and set the output vector 
        for i in range(self.n_layers-1):
            y_next = self.__forward(self.layers[i], self.layers[i+1])
         
            if i+1 == self.n_layers-1:
                self.layers[i+1].set_output_vector(y_next)
            else:
                self.layers[i+1].set_input_vector(y_next)

        return self.layers[-1].get_output_vector()
            
