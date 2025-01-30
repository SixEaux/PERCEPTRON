import numpy as np


class Layer:
    def __init__(self, nbneurons, activation, nbinputs, difffct):
        self.weights = np.ones((nbinputs, nbneurons))  # matrix last layers number of outputs * nb of neurons
        self.biases = np.zeros((1, nbneurons))

        self.activation = activation
        self.difactivation = difffct

    def forward(self, input):
        return self.activation(np.dot(input, self.weights)) + self.biases

    def backward(self, erroroutput):
        error = np.dot(self.weights.T, erroroutput)