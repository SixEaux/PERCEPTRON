import numpy as np

class Layer:
    def __init__(self, nbneurons, acti, lnginputs):

        self.weights = np.random.uniform(-1, 1, (lnginputs, nbneurons))
        self.bias = np.random.uniform(-1, 1, (nbneurons, 1))

        if acti == 'sigmoid':
            self.activation =  lambda x: 1 / (1 + np.exp(-x))
            self.difactivation = lambda x: np.exp(-x) / (1 + np.square(np.exp(-x)))

        elif acti == 'relu':
            self.activation =  lambda x: np.where(x > 0, x, 0)
            self.difactivation = lambda x: np.where(x > 0, x, 0)

        elif acti == 'tanh':
            self.activation =  lambda x: np.tanh(x)
            self.difactivation = lambda x: 1 - np.square(np.tanh(x))

        else:
            pass

    def forward(self, input):
        return self.activation(np.dot(np.transpose(self.weights), input) + self.bias)

    def backward(self, diffnextlayer):
        pass


class OutputLayer(Layer):
    def __init__(self, nbneurons, activation, lnginputs):
        super().__init__(nbneurons, activation, lnginputs)

    def softmax(self, input):
        return np.exp(input) / np.sum(np.exp(input), axis=0)

    def softmaxdiff(self, output):
        pass

    def forward(self, input):
        return self.softmax(np.dot(np.transpose(self.weights), input) + self.bias)

    def backward(self, diffnextlayer):
        pass
