import numpy as np
import pickle
from tabulate import tabulate
from LAYER import Layer, OutputLayer


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

        self.createlayers()

    def printbasesimple(self, base):
        print(tabulate(base.reshape((28,28))))

    def createlayers(self): #create all layers
        for i in range(self.nblay):
            self.layers.append(Layer(*self.infolay[i]))
        self.layers.append(OutputLayer(10, "eqm", self.infolay[-1][0]))

    def forwardprop(self, input): #forward all the layers until output
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
        return input

    def backprop(self, observed, expected):
        pass

    def trainsimple(self):
        pass

    def trainbatch(self):
        pass

    def tauxerreur(self): #go in all the test and see accuracy
        pass


val, pix = takeinputs()
lay = [[64, "sigmoid", 784]]

g = NN(pix, val, 1, lay, "eqm")


l = g.forwardprop((pix[10].reshape(784,1))/255)
print(l)
