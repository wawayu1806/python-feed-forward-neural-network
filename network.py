from layer import Layer

class Network:
    def __init__(self, n_layers=2):
        self.n_layers = n_layers 
        self.layers = [] 

    def initalize_network(self, layers):
        for i in layers:
            self.layers.append(Layer(i))

        self.n_layers = len(self.layers)
        
    def connect_layers(self):
        for i in range(self.n_layers-1):
            self.layers[i].initialize_weight_matrix(self.layers[i+1].n_neurons)