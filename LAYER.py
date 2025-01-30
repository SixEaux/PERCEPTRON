import numpy as np


class Layer:
    def __init__(self, nbneurons, activation, nbinputs, difffct):
        self.weights = np.ones((nbinputs, nbneurons))  # matrix last layers number of outputs * nb of neurons
        self.biases = np.zeros((1, nbneurons))

        self.activation = self.getactivationfunct(activation)
        self.difactivation = self.getactivationdiff(activation)

    def forward(self, input):
        return self.activation(np.dot(input, self.weights)) + self.biases

    def backward(self, erroroutput):
        error = np.dot(self.weights.T, erroroutput)

    @property
    def getactivationfunct(self, activation):
        if activation == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))

        elif activation == 'relu':
            return lambda x: x > 0

        elif activation == 'tanh':
            return lambda x: np.tanh(x)

    @property
    def getactivationdiff(self, activation):
        if activation == 'sigmoid':
            return lambda x: np.exp(-x) / (1 + np.square(np.exp(-x)))

        elif activation == 'relu':
            return lambda x: x > 0

        elif activation == 'tanh':
            return lambda x: 1 - np.square(np.tanh(x))



