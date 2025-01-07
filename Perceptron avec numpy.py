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
    def __init__(self, pix, vales, *, nbneurones = 784, coefcv = 0.1, iterations=1000, seuil = 0, normal = False,
                 bruitgaussien = False, pourcecarttype = 0,
                 bruitsurpix = False, nbpixelsbruit = 0, positionschoisies = False):

        self.iter = iterations #nombre iteration entrainement
        self.nb = nbneurones #nombre de neurones

        #INITIALISATION VARIABLES
        self.poids = np.ones(nbneurones + 1) #np.random.randn(nbneurones + 1) * 0.01 #avec le biai qui est la premiere valeur
        self.cvcoef = coefcv
        self.seuil = seuil
        self.biais = 1

        #INPUTS POUR ENTRAINEMENT
        self.pix = pix
        self.vales = vales


        #BRUIT GAUSSIEN
        self.normal = normal
        self.boolbruitgaus = bruitgaussien
        self.pix = self.normaliserbase(self.pix)
        self.ecarttype = pourcecarttype / 100

        #BRUIT PAR PIXELS
        self.boolbruitpix = bruitsurpix
        self.choisies = positionschoisies
        self.nbbruit = nbpixelsbruit


    def normaliserbase(self, base):
        return base/255

    def bruitgaussien(self, image):
        bruit = np.random.normal(0, self.ecarttype, 784)
        imagebrouillee = np.clip(image + bruit, 0, 1) #je ne sais pas si nécessaire mais pour qu'il n'y ait pas de valeur au de la
        return imagebrouillee

    def bruitpixels(self, image, norm, saturation):
        sat = (1,255) if saturation else (0,0)
        if self.choisies:
            try:
                posbruits = list(map(int,list(input(f"Donnez moi une liste avec les positions que vous voulez modifier \n"
                                                    f" nombres entre 0 et 783 et separes par des virgules: ").split(","))))
                if len(posbruits) > self.nbbruit:
                    posbruits = posbruits[:self.nbbruit]
                print(posbruits)
            except:
                print("Ce n'est pas le format correct donc pas de bruit")
                posbruits = []
        else:
            posbruits = random.sample(range(0, 783), self.nbbruit)

        for el in posbruits:
            image[el] = sat[0] if norm else sat[1]

        return image

    def printbasesimple(self, base):
        print(tabulate(base[1:].reshape((28,28))))

    def printcouleur(self, base):
        df2 = base[1:].reshape((28,28))
        plt.imshow(df2, cmap='Greys', interpolation='nearest')
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

    def testuneimage(self, image, recherch):
        self.printcouleur(self.poids)
        self.printcouleur(self.vraiinput(image))
        predator = self.prediction(image)
        if predator == 1:
            print(f"À mon avis ce chiffre est un {recherch}")
        else:
            print(f"À mon avis ce chiffre ne ressemble point à un {recherch}")



P = Perceptron(pixels, valeurs, coefcv = 0.2, seuil = 0, normal=True,
               bruitgaussien = False, pourcecarttype = 5,
               bruitsurpix = False, nbpixelsbruit = 765, positionschoisies = True)

P.entrainementun(0)

im = P.bruitpixels(qcmpix[3], False, True)

P.testuneimage(im, 0)

# print(P.tauxerreur(0, qcmpix, qcmval))
