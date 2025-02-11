import numpy as np
from Functions import SoftMax

class Layer:
    def __init__(self, nbneurons, activation, lnginputs):
        self.weights = np.ones((lnginputs, nbneurons))
        self.bias = np.zeros((nbneurons, 1))

        self.activation = self.getactivationfunct(activation)
        self.difactivation = self.getactivationdiff(activation)

    @property
    def getactivationfunct(self, activation):
        if activation == 'softmax':
            return lambda z: np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        elif activation == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))

        elif activation == 'relu':
            return lambda x: x > 0

        elif activation == 'tanh':
            return lambda x: np.tanh(x)

    @property
    def getactivationdiff(self, activation):
        if activation == 'softmax':
            return lambda z: np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        elif activation == 'sigmoid':
            return lambda x: np.exp(-x) / (1 + np.square(np.exp(-x)))

        elif activation == 'relu':
            return lambda x: x > 0

        elif activation == 'tanh':
            return lambda x: 1 - np.square(np.tanh(x))

    def backward(self, diffnextlayer):
        pass

    def forward(self, input):
        return self.activation(np.dot(self.weights.T, input) + self.bias)

class OutputLayer(Layer):
    def __init__(self, nbneurons, activation, lnginputs):
        super().__init__(nbneurons, activation, lnginputs)


    def backward(self, diffnextlayer):
        pass


