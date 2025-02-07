import numpy as np

class ReLu:

    def forward(self, input):
        return input > 0

    def backward(self, input):
        return input > 0

class Sigmoide:

    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input):
        return self.forward(input) * (1 - self.forward(input))

class SoftMax:

    def forward(self, input):
        return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)

    def backward(self, input):
        return

class Error:

    def CCE(self, output, indexbonoutput): #CategoricalCrossEntropy: erreur calculée avec un vecteur avec 0 partout en 1 à la position voulu et l'erreur est juste -ln() de la position dans l'output où il y a le 1
        return -np.log(output[indexbonoutput])

    def backward(self, input):
        pass