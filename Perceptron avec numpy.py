import numpy as np
import pickle
import random
from tabulate import tabulate
import matplotlib.pyplot as plt

with open('valeursentraine', 'rb') as f:
    valeurs = np.array(pickle.load(f))
    vali = np.array(valeurs[:10000])

with open('pixelsentraine', 'rb') as f:
    pixels = np.array(pickle.load(f))
    pixi = np.array(pixels[:10000])

with open('testval', 'rb') as f:
    qcmval = np.array(pickle.load(f))

with open('testpix', 'rb') as f:
    qcmpix = np.array(pickle.load(f))

class Perceptron:
    def __init__(self, nbneurones, pix, vales, *, coefcv = 0.1, iterations=1000, seuil = 0, normal = False,
                 bruitgaussien = False, pourcecarttype = 0,
                 bruitsurpix = False, nbpixelsbruit = 0, positionschoisies = False, minmaxchangementpix = (0, 0)):

        self.iter = iterations
        self.nb = nbneurones

        self.poids = np.ones(nbneurones + 1) #np.random.randn(nbneurones + 1) * 0.01 #avec le biai qui est la premiere valeur
        self.cvcoef = coefcv
        self.seuil = seuil
        self.biais = 1

        self.pix = pix
        self.vales = vales

        self.normal = normal
        self.boolbruitgaus = bruitgaussien

        if self.normal:
            self.pix = self.normaliserbase(self.pix)
            if bruitgaussien:
                self.ecarttype = pourcecarttype / 100

        if nbpixelsbruit > 0:
            self.nbbruit = nbpixelsbruit
        self.choisies = positionschoisies
        self.changementpix = minmaxchangementpix


    def normaliserbase(self, base):
        return base/255

    def bruitgaussien(self, image):
        bruit = np.random.normal(0, self.ecarttype, 784)
        imagebrouillee = np.clip(image + bruit, 0, 1) #je ne sais pas si nécessaire mais pour qu'il n'y ait pas de valeur au de la
        return imagebrouillee

    def bruitpixels(self, image):
        if self.choisies:
            try:
                posbruits = list(input("Donnez moi une liste avec les positions que vous voulez modifier: "))
                if len(posbruits) > self.nbbruit:
                    posbruits = posbruits[:self.nbbruit]
            except:
                print("Ce n'est pas une liste donc pas de bruit")
                posbruits = []
        else:
            posbruits = [np.random.randint(0,784) for i in range(self.nbbruit)]

        for el in posbruits:
            image[el] += np.random.randint(self.changementpix[0], self.changementpix[1])

    def printbasesimple(self, base):
        print(tabulate(base[1:].reshape((28,28))))

    def printcouleur(self, base):
        df2 = base[1:].reshape((28,28))
        plt.imshow(df2, cmap='Dark2', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.title("Array Visualization")
        plt.show()

    def fctactivescalier(self, x):
        return 1 if x>self.seuil else 0

    def produit(self,tab1,tab2):
        return np.dot(tab1,tab2)

    def vraiinput(self, input):
        return np.concatenate(([self.biais],input))

    def changerpoids(self, attendu, observe, input):
        vrainput = self.vraiinput(input)
        self.poids += self.cvcoef * (attendu - observe) * vrainput

    def validation(self, recherchee, valeur):
        return 1 if recherchee == valeur else 0

    def prediction(self, input):
        return self.fctactivescalier(self.produit(self.poids, self.vraiinput(input)))

    def entrainementun(self, recherch):
        for fig in range(len(self.pix)):
            pred = self.prediction(self.pix[fig])
            self.changerpoids(self.validation(recherch, self.vales[fig]), pred, self.pix[fig])

    def tauxerreur(self, recherch, basepix, baseval):
        if not self.normal:
            correct = 0
            for i in range(len(basepix)):
                predator = self.prediction(basepix[i])
                correct += 1 if predator == self.validation(recherch, baseval[i]) else 0
            return 100 - correct*100/len(basepix)
        else:
            basepix = self.normaliserbase(basepix)
            correct = 0
            for i in range(len(basepix)):
                if self.boolbruitgaus:
                    predator = self.prediction(self.bruitgaussien(basepix[i]))
                else:
                    predator = self.prediction(basepix[i])
                correct += 1 if predator == self.validation(recherch, baseval[i]) else 0
            return 100 - correct * 100 / len(basepix)

P = Perceptron(784, pixels, valeurs, coefcv = 0.2, seuil = 0, normal=True)

P.entrainementun(0)
P.printbasesimple(P.poids)

print(P.tauxerreur(0, qcmpix, qcmval))
