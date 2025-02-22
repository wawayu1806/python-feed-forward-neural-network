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
            self.layers.append(HiddenLayer(i))

        # add the output layer and update self.n_layers
        outputlayer = OutputLayer(layers[-1])
        self.layers.append(outputlayer)
        self.n_layers = len(self.layers) 

    def connect_layers(self):
        for i in range(self.n_layers-1):
            self.layers[i].initialize_weight_matrix(self.layers[i+1].n_neurons)