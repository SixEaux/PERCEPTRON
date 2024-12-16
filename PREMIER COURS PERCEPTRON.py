import numpy as np
import pickle
import random
from tabulate import tabulate
import matplotlib.pyplot as plt

with open('valeursentraine', 'rb') as f:
    valeurs = pickle.load(f)
    vali = valeurs[:10000]

with open('pixelsentraine', 'rb') as f:
    pixels = pickle.load(f)
    pixi = pixels[:10000]

with open('testval', 'rb') as f:
    qcmval = pickle.load(f)

with open('testpix', 'rb') as f:
    qcmpix = pickle.load(f)

class ImageReader:
    def __init__(self):
        self.rng = np.random.default_rng()

    def fakeImg(self, size):
        image = self.rng.integers ( 0 ,  255 , size*size)
        image.reshape((size,size))
        return image


class Perceptron:
    def __init__(self, nbneurones, pix, vales, *, coefcv = 0.1, iterations=1000, seuil = 0, normal = False):
        self.iter = iterations
        self.nb = nbneurones
        self.poids = [1 for _ in range(nbneurones)]
        self.cvcoef = coefcv
        self.seuil = seuil
        self.biais = 1
        self.pix = pix
        self.vales = vales
        if normal:
            self.normaliserbase(self.pix)
        print(self.pix[0])

    def normaliserbase(self, base):
        p = [[j/255 for j in i] for i in base]


    def autreautreprint(self, lista):
        df = np.array(lista, copy=True).reshape((28, 28))
        print(tabulate(list(df)))


    def autreautreautreprint(self, lista):
        df = np.array(lista, copy=True).reshape((28, 28))
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
        return 1 if activ == 1 else 0


    def entrainementun(self, recherch):#erreur de penser
        for fig in range(len(self.pix)):
            pred = self.prediction(self.pix[fig])
            self.changerpoids(self.validation(recherch, self.vales[fig]), pred, self.pix[fig])


    def tauxerreur(self, recherch, basepix, baseval):
        correct = 0
        for i in range(len(basepix)):
            predator = self.prediction(basepix[i])
            if predator == self.validation(recherch, baseval[i]):
                correct += 1
        return 100 - correct*100/len(basepix)


P = Perceptron(784, pixi, vali, coefcv = 0.1, seuil = 0, normal=False)

P.entrainementun(8)
P.autreautreautreprint(P.poids)

print(P.tauxerreur(8, qcmpix, qcmval))

