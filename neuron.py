import numpy as np

class Neuron:

    @staticmethod
    def sigmoid(x):  
        return 1/(1+np.exp(-x))

    def __init__(self, b):
        self.b = b

    def set_bias(self, b):
        self.b = b

    def get_bias(self):
        return self.b