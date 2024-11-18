import numpy as np
import pickle
import pprint
import random


with open('valeursentraine', 'rb') as f:
    valeurs = pickle.load(f)

with open('pixelsentraine', 'rb') as f:
    pixels = pickle.load(f)
# print(pixels[0])
# print(len(pixels[0]))
# print(len(pixels))

class ImageReader():
    def __init__(self):
        self.rng = np.random.default_rng()

    def fakeImg(self, size):
        image = self.rng.integers ( 0 ,  255 , size*size)
        image.reshape((size,size))
        return image




class Perceptron:
    def __init__(self, nbneurones, pix, vales, *, coefcv = 0.1, iterations=1000, seuil = 0):
        self.iter = iterations
        self.nb = nbneurones
        self.poids = [1 for _ in range(nbneurones)]
        self.cvcoef = coefcv
        self.seuil = seuil
        self.biais = 1
        self.pix = pix
        self.vales = vales

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

    def changerpoids(self, attendu, observation, input):
        #si bonne reponse on garde les poids, si erreur pensant que c'est le chiffre attendu - reponse = -1 sinon inverse = 1
        vrainput = self.vraiinput(input)
        for i in range(1, len(self.poids)):
            self.poids[i] = self.cvcoef * (attendu - observation) * vrainput[i]
        self.poids[0] += self.cvcoef * (attendu - observation)


    def vraivaleur(self, recherchee, valeur):
        return 1 if recherchee == valeur else 0

    def vraiinput(self, input):
        return [self.biais] + input

    def prediction(self, input):
        vrainput = self.vraiinput(input)
        attendu = self.produit(self.poids, vrainput)
        activ = self.fctactiv(attendu)
        return 1 if activ == 1 else 0

    def entrainementuns(self):#erreur de penser
        for fig in range(len(self.pix)):
            pred = self.prediction(self.pix[fig])
            self.changerpoids(self.vraivaleur(1, self.vales[fig]), pred, self.pix[fig])

    def tauxerreur(self):
        correct = 0
        for i in range(len(self.pix)):
            predator = self.prediction(self.pix[i])
            if predator == self.vraivaleur(1, self.vales[i]):
                correct += 1
        return 100 - correct*100/len(self.pix)


P = Perceptron(784, pixels, valeurs)

print(P.poids)
P.entrainementuns()
print(P.poids)

print(P.tauxerreur())

