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
    def __init__(self, pix, vales, infolay, errorfunc, *, coefcv=0.1, iterations=10):
        self.iter = iterations  # nombre iteration entrainement
        self.nblay = len(infolay) # nombre de layers

        # INITIALISATION VARIABLES
        self.cvcoef = coefcv

        # INPUTS POUR ENTRAINEMENT
        self.pix = pix/255
        self.vales = vales

        self.parameters = self.params(infolay)

        self.errorfunc = self.geterrorfunc(errorfunc)

    def printbasesimple(self, base):
        print(tabulate(base.reshape((28, 28))))

    def geterrorfunc(self, errorfunc): #exp est un onehotvect
        if errorfunc == "eqm":
            return [lambda obs, exp, nbinput: (np.sum((obs - exp) ** 2, axis=1)) / (2 * nbinput), lambda obs, expected: (obs - expected)]
        elif errorfunc == "CCC":
            return [lambda obs, expected: -np.sum(expected * np.log(np.clip(obs, 1e-7, 1 - 1e-7)), axis=1)] #si le exp c'est un one hot ver¡cteurs
            # si place bon output: return lambda obs, exp: -np.log(np.clip(obs, 1e-7, 1 - 1e-7)[exp, 1])
            # il manque la diff

    def getfct(self, acti):
        if acti == 'sigmoid':
            return [lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(-x) / (1 + np.square(np.exp(-x)))]

        elif acti == 'relu':
            return [lambda x: np.where(x > 0, x, 0), lambda x: np.where(x > 0, x, 0)]

        elif acti == 'tanh':
            return [lambda x: np.tanh(x), lambda x: 1 - np.square(np.tanh(x))]

        elif acti == 'softmax':
            return [lambda input: np.exp(input) / np.sum(np.exp(input), axis=0), None]

        else:
            pass

    def params(self, lst): #lst liste avec un tuple avec (nbneurons, fctactivation)
        param = {}



        for l in range(1, len(lst)):
            param["w" + str(l)] = np.random.uniform(-1,1,(lst[l-1][0], lst[l][0]))
            param["b" + str(l)] = np.random.uniform(-1,1,(lst[l][0], 1))
            param["fct" + str(l)] = self.getfct(lst[l][1])[0]
            param["diff" + str(l)] = self.getfct(lst[l][1])[1]

        return param


    def forwardprop(self, input): #forward all the layers until output
        outlast = input
        vieux = [] #garder pour la backprop les variables
        for l in range(1, self.nblay):
            activavant = outlast
            w = self.parameters["w" + str(l)]
            b = self.parameters["b" + str(l)]
            z = np.dot(w.T, activavant) + b
            a = self.parameters["fct" + str(l)](z)
            vieux.append((activavant,w,b,z))
            outlast = a

        return outlast, vieux


    def backprop(self, observed, expected):
        pass

    def trainsimple(self):
        pass

    def trainbatch(self):
        pass

    def tauxerreur(self): #go in all the test and see accuracy
        pass


val, pix = takeinputs()
lay = [(784,"input"), (64,"relu"), (20, "sigmoid"), (10, "softmax")]

g = NN(pix, val, lay, "eqm")


l = g.forwardprop((pix[10].reshape(784,1))/255)
print(l[0])
