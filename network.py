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

    def feed_forward(self, input_data):
        self.layers[0].set_input_vector(input_data)

        for i in range(self.n_layers):
            output = self.layers[i].forward()
            
            if i < self.n_layers - 1:
                self.layers[i+1].set_input_vector(output)

        return output