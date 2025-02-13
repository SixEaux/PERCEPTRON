import numpy as np
import pickle
from tabulate import tabulate


# Plus tard:
# do method with batchs to do it directly with 32 for example
# le input est un batch en forme de matrice avec 32 lignes et 784 colonnes
# expected will be a one hot vector


def takeinputs():

    with open('valeursentraine', 'rb') as f:
        valeurs = np.array(pickle.load(f))
        vali = np.array(valeurs[:10000])

    with open('pixelsentraine', 'rb') as f:
        pixels = np.array(pickle.load(f))
        pixi = np.array(pixels[:10000])

    with open('testval', 'rb') as f:
        qcmval = np.array(pickle.load(f))
        petitqcmval = np.array(qcmval[0:5000])

    with open('testpix', 'rb') as f:
        qcmpix = np.array(pickle.load(f))
        petitqcmpix = np.array(qcmpix[0:5000])

    return valeurs, pixels


class Layer:
    def __init__(self, nbneurons, acti, lnginputs):

        self.weights = np.ones((lnginputs, nbneurons))
        self.bias = np.ones((nbneurons, 1))

        if acti == 'sigmoid':
            self.activation =  lambda x: 1 / (1 + np.exp(-x))
            self.difactivation = lambda x: np.exp(-x) / (1 + np.square(np.exp(-x)))

        elif acti == 'relu':
            self.activation =  lambda x: np.where(x > 0, x, 0)
            self.difactivation = lambda x: np.where(x > 0, x, 0)

        elif acti == 'tanh':
            self.activation =  lambda x: np.tanh(x)
            self.difactivation = lambda x: 1 - np.square(np.tanh(x))

    def forward(self, input):
        return self.activation(np.dot(np.transpose(self.weights), input) + self.bias)

    def backward(self, diffnextlayer):
        pass


class OutputLayer(Layer):
    def __init__(self, nbneurons, activation, lnginputs):
        super().__init__(nbneurons, activation, lnginputs)

    def softmax(self, input):
        return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)

    def softmaxdiff(self, output):
        pass

    def forward(self, input):
        return self.softmax(np.dot(np.transpose(self.weights), input) + self.bias)

    def backward(self, diffnextlayer):
        pass

class NN:
    def __init__(self, pix, vales, nblayer, infolay, errorfunc, *, coefcv=0.1, iterations=10):
        self.iter = iterations  # nombre iteration entrainement
        self.nblay = nblayer # nombre de layers

        # INITIALISATION VARIABLES
        self.cvcoef = coefcv

        # INPUTS POUR ENTRAINEMENT
        self.pix = pix/255
        self.vales = vales

        self.infolay = infolay # list de dictionnaires avec les params de chaque layer sauf input et output
        self.layers = []

        self.errorfunc = self.geterrorfunc(errorfunc)

        self.errordiff = self.geterrordiff(errorfunc)

        self.createlayers()

    def printbasesimple(self, base):
        print(tabulate(base.reshape((28,28))))

    def createlayers(self): #create all layers
        for i in range(self.nblay):
            self.layers.append(Layer(*self.infolay[i]))
        self.layers.append(OutputLayer(10, None, self.infolay[-1][0]))

    def geterrorfunc(self, errorfunc): #exp est un onehotvect
        if errorfunc == "eqm":
            return lambda obs, exp: np.sum((obs-exp)**2, axis=1)
        elif errorfunc == "CCC":
            return lambda obs, exp: -np.sum(exp * np.log(np.clip(obs, 1e-7, 1 - 1e-7)), axis=1) #si le exp c'est un one hot ver¡cteurs
            # si place bon output: return lambda obs, exp: -np.log(np.clip(obs, 1e-7, 1 - 1e-7)[exp, 1])

    def geterrordiff(self, errorfunc):
        if errorfunc == "eqm":
            return
        elif errorfunc == "CCC":
            return

    def forwardprop(self, input): #forward all the layers until output
        for i in range(len(self.layers)):
            res = self.layers[i].forward(input)
            print(res)
            input = res
        return input

    def backprop(self, observed, expected):
        error = self.errorfunc(observed, expected)


    def train(self):
        pass

    def tauxerreur(self): #go in all the test and see accuracy
        pass


val, pix = takeinputs()
lay = [[64, "relu", 784]]

g = NN(pix, val, 1, lay, "eqm")


g.printbasesimple(pix[10].reshape(784,1))
l = g.forwardprop((pix[10].reshape(784,1))/255)
print(l.shape)
