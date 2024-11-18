import numpy as np
import pickle
import pprint


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
    def __init__(self,nbneurones, *, coefcv = 0.1, iterations=1000, seuil = 0):
        self.iter = iterations
        self.nb = nbneurones
        self.poids = [1 for _ in range(nbneurones+1)]
        self.cvcoef = coefcv
        self.seuil = seuil
        self.biais = 1


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
            print(fx)
            return fx
        except TypeError:
            print(tab1,"\n", tab2)
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

    def entrainementuns(self, pix, val):#erreur de penser
        for fig in range(len(pix)):
            pred = self.prediction(pix[fig])
            self.changerpoids(self.vraivaleur(1, val[fig]), pred, pix[fig])


P = Perceptron(784)
P.entrainementuns(pixels,valeurs)



def cherche1(tab):
    for i in range(len(tab)):
        if tab[i] == 1:
            return i
a = cherche1(valeurs)
print(P.prediction(pixels[a]))
print(P.vraivaleur(1, valeurs[a]))

