import numpy as np


class ImageReader():
    def __init__(self):
        self.rng = np.random.default_rng()

    def fakeImg(self, size):
        image = self.rng.integers ( 0 ,  255 , size*size)
        image.reshape((size,size))
        return image



class Perceptron:
    def __init__(self,nbneurones, coefcv = 0.1, iterations=1000, sup = 0):
        self.iter = iterations
        self.nb = nbneurones
        self.neurones = [None for i in range(nbneurones)]
        self.poids = [1 for i in range(nbneurones)]
        self.cvcoef = coefcv
        self.sup = sup


    def fctactiv(self, x):
        if x>self.sup:
            return 1
        else:
            return 0

    def produit(self, tab1,tab2):
        fx = 0
        for i in range(len(tab1)):
            fx += tab1[i]*tab2[i]
        return fx


    def changerpoids(self, attendu, observation, input):
        #si bomne reponse on garde les poids, si erreur pensant que c'est le chiffre attendu - reponse = -1 sinon inverse = 1
        for i in range(len(self.poids)):
            self.poids[i] = self.cvcoef * (attendu[i] - observation[i]) * input[i]


    def prediction(self, input):
        attendu = self.produit(self.poids, input)
        activ = self.fctactiv(attendu)
        return 1 if activ > 1 else 0



