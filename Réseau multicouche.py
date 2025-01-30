import numpy as np
import pickle

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

class fct:
    def __init__(self):
        self.sig = lambda x: 1/(1+np.exp(-x))
        self.difsig = lambda x: (1/(1+np.exp(-x))) * (1-(1/(1+np.exp(-x))))


class NN(fct):
    def __init__(self, pix, vales, nblayer, *, nbneurones=784, coefcv=0.1, iterations=10, seuil=0.0):
        super().__init__()
        self.iter = iterations  # nombre iteration entrainement
        self.nbneur = nbneurones  # nombre de neurones
        self.nblay = nblayer # nombre de layers

        # INITIALISATION VARIABLES
        self.poids = np.ones(nbneurones + 1)  # np.random.randn(nbneurones + 1) * 0.01 #avec le biai qui est la premiere valeur
        self.cvcoef = coefcv
        self.seuil = seuil
        self.biais = 1

        # INPUTS POUR ENTRAINEMENT
        self.pix = pix/255
        self.vales = vales

    def produit(self, tab1, tab2):
        return np.dot(tab1, tab2)

    def forward(self):
        pass

    def backward(self):
        pass




