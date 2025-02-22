from network import Network
import numpy as np

def main():
    network = Network()
    network.initalize_network([3,4,3])
    network.connect_layers()
    network.feed_forward(np.array([1,2,3]))

if __name__ == "__main__":
    main()