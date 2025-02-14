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

    def forward(self, input): # ca peut se faire direct mais comme ca clair
        z = np.dot(np.transpose(self.weights), input) + self.bias
        a = self.activation(z)
        return a

    def backward(self, errornextlayer, zlayer):
        errorlayer = np.multiply(np.dot(np.transpose(self.weights), errornextlayer), self.difactivation(zlayer))


class OutputLayer(Layer):
    def __init__(self, nbneurons, errorfunc, lnginputs):
        super().__init__(nbneurons, "NON", lnginputs)

        self.errorfunc = self.geterrorfunc(errorfunc)

        self.errordiff = self.geterrordiff(errorfunc)

    def softmax(self, input):
        return np.exp(input) / np.sum(np.exp(input), axis=0)

    def softmaxdiff(self, output):
        pass

    def geterrorfunc(self, errorfunc): #exp est un onehotvect
        if errorfunc == "eqm":
            return lambda obs, exp, nbinput: (np.sum((obs-exp)**2, axis=1))/(2 * nbinput)
        elif errorfunc == "CCC":
            return lambda obs, expected: -np.sum(expected * np.log(np.clip(obs, 1e-7, 1 - 1e-7)), axis=1) #si le exp c'est un one hot ver¡cteurs
            # si place bon output: return lambda obs, exp: -np.log(np.clip(obs, 1e-7, 1 - 1e-7)[exp, 1])

    def geterrordiff(self, errorfunc):
        if errorfunc == "eqm":
            return lambda obs, expected: (obs-expected)
        elif errorfunc == "CCC":
            return

    def forward(self, input):
        return self.softmax(np.dot(np.transpose(self.weights), input) + self.bias)

    def backward(self, diffnextlayer):
        pass
