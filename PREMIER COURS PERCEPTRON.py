import numpy as np
import pickle
import random
from tabulate import tabulate
import matplotlib.pyplot as plt

with open('Datas/valeursentraine', 'rb') as f:
    valeurs = pickle.load(f)
    vali = valeurs[:10]

with open('Datas/pixelsentraine', 'rb') as f:
    pixels = pickle.load(f)
    pixi = pixels[:10]

with open('Datas/testval', 'rb') as f:
    qcmval = pickle.load(f)

with open('Datas/testpix', 'rb') as f:
    qcmpix = pickle.load(f)

class ImageReader:
    def __init__(self):
        self.rng = np.random.default_rng()

    def fakeImg(self, size):
        image = self.rng.integers ( 0 ,  255 , size*size)
        image.reshape((size,size))
        return image


class Perceptron:
    def __init__(self, nbneurones, pix, vales, *, coefcv = 0.1, iterations=1000, seuil = 0, normal = True):
        self.iter = iterations
        self.nb = nbneurones
        self.poids = [1 for _ in range(nbneurones + 1)]
        self.cvcoef = coefcv
        self.seuil = seuil
        self.biais = 1
        self.pix = pix
        self.vales = vales
        self.normal = normal
        if normal:
            self.pix = self.normaliserbase(self.pix)

    def normaliserbase(self, base):
        return [[j/255 for j in i] for i in base]


    def autreautreprint(self, lista):
        df = np.array(lista[1:], copy=True).reshape((28, 28))
        print(tabulate(list(df)))


    def autreautreautreprint(self, lista):
        df = np.array(lista[1:], copy=True).reshape((28, 28))
        df2 = np.where(df>5, 1, 0)
        plt.imshow(df2, cmap='Dark2', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.title("Array Visualization")
        plt.show()


    def fctactiv(self, x):
        if x>self.seuil:
            return 1
        else:
            return 0

    def produit(self, tab1,tab2):
        try:
            fx = 0
            for i in range(len(tab1)):
                fx += tab1[i]*tab2[i]
            return fx
        except TypeError:
            print("oh non erreur", tab1,"\n", tab2)
            return

    def vraiinput(self, input):
        return [self.biais] + input

    def changerpoids(self, attendu, observation, input):
        #si bonne reponse on garde les poids, si erreur pensant que c'est le chiffre attendu - reponse = -1 sinon inverse = 1
        vrainput = self.vraiinput(input)
        for i in range(1, len(self.poids)):
            self.poids[i] += self.cvcoef * (attendu - observation) * vrainput[i]
        self.poids[0] += self.cvcoef * (attendu - observation)

    def validation(self, recherchee, valeur):
        return 1 if recherchee == valeur else 0

    def prediction(self, input):
        vrainput = self.vraiinput(input)
        attendu = self.produit(self.poids, vrainput)
        activ = self.fctactiv(attendu)
        return activ

    def entrainementun(self, recherch):#erreur de penser
        for fig in range(len(self.pix)):
            pred = self.prediction(self.pix[fig])
            self.changerpoids(self.validation(recherch, self.vales[fig]), pred, self.pix[fig])

    def tauxerreur(self, recherch, basepix, baseval):
        if not self.normal:
            correct = 0
            for i in range(len(basepix)):
                predator = self.prediction(basepix[i])
                if predator == self.validation(recherch, baseval[i]):
                    correct += 1
            return 100 - correct*100/len(basepix)
        else:
            basepix = self.normaliserbase(basepix)
            correct = 0
            for i in range(len(basepix)):
                predator = self.prediction(basepix[i])
                if predator == self.validation(recherch, baseval[i]):
                    correct += 1
            return 100 - correct * 100 / len(basepix)


P = Perceptron(784, pixels, valeurs, coefcv = 0.1, seuil = 0)

P.entrainementun(1)
# P.autreautreautreprint(P.poids)

print(P.tauxerreur(1, qcmpix, qcmval))

