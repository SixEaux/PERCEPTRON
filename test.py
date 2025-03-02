import numpy as np
import pickle
from scipy.special import expit
# from itertools import batched

from tabulate import tabulate

# Plus tard:
# do method with batchs to do it directly with 32 for example
# le input est un batch en forme de matrice avec 784 lignes et 32 colonnes
# expected will be a one hot vector

np.seterr(all='raise')

def takeinputs():

    with open('valeursentraine', 'rb') as f:
        valeurs = np.array(pickle.load(f))
        vali = np.array(valeurs[0:40000])

    with open('pixelsentraine', 'rb') as f:
        pixels = np.array(pickle.load(f))
        pixi = np.array(pixels[0:40000])

    with open('testval', 'rb') as f:
        qcmval = np.array(pickle.load(f))
        petitqcmval = np.array(qcmval[0:50])

    with open('testpix', 'rb') as f:
        qcmpix = np.array(pickle.load(f))
        petitqcmpix = np.array(qcmpix[0:50])

    return valeurs, pixels, qcmval, qcmpix

class NN:
    def __init__(self, pix, vales, infolay, errorfunc, qcmpix, qcmval, *, coefcv=0.1, iterations=1, batch=1):
        self.iter = iterations  # nombre iteration entrainement
        self.nblay = len(infolay)-1 # nombre de layers

        # INITIALISATION VARIABLES
        self.cvcoef = coefcv

        # INPUTS POUR ENTRAINEMENT
        self.pix = self.processdata(pix) #pix de train
        print(self.pix.shape)
        self.vales = vales #val de train

        self.qcmpix = qcmpix
        self.qcmval = qcmval

        self.parameters = self.params(infolay) #creer les parametres dans un dico/ infolay doit avoir tout au debut la longueur de l'input

        self.errorfunc = self.geterrorfunc(errorfunc)[0] #choisir la fonction d'erreur
        self.differrorfunc = self.geterrorfunc(errorfunc)[1]

        self.lenbatch = batch

    def printbasesimple(self, base):
        print(tabulate(base.reshape((28, 28))))

    def processdata(self, pix): #mettre les donnees sous la bonne forme
        data = pix/255
        print(data[:5])
        datamod = np.array([np.array(a).reshape(784,1) for a in data])
        return datamod

    def params(self, lst): #lst liste avec un tuple avec (nbneurons, fctactivation)
        param = {}

        for l in range(1, len(lst)):
            param["w" + str(l-1)] = np.random.rand(lst[l][0], lst[l-1][0]) - 0.5
            #np.random.randn(lst[l][0], lst[l-1][0]) * np.sqrt(2 / lst[l-1][0])
            # #np.random.uniform(-1, 1, (lst[l][0], lst[l-1][0])) #nbneurons * nbinput
            param["b" + str(l-1)] = np.random.rand(lst[l][0], 1) - 0.5 #np.zeros((lst[l][0], 1))
            param["fct" + str(l-1)] = self.getfct(lst[l][1])[0]
            param["diff" + str(l-1)] = self.getfct(lst[l][1])[1]
        return param

    def geterrorfunc(self, errorfunc): #exp est un onehotvect
        if errorfunc == "eqm":
            def eqm(obs, exp, nbinput=1):
                return (np.sum((obs - exp) ** 2, axis=1))/ (2 * nbinput)
            def eqmdif(obs, expected, nbinput=1):
                return  (obs - expected)/nbinput
            return [eqm, eqmdif]

        elif errorfunc == "CEL":
            def CEL(obs, exp, nbinput=1):
                return -np.sum(exp * np.log(np.clip(obs, 1e-9, 1 - 1e-9)), axis=1) / nbinput
            def CELdif(obs, exp, nbinput=1):
                return (obs - exp) / nbinput
            return [CEL, CELdif]

    def getfct(self, acti):
        if acti == 'sigmoid':
            def sigmoid(x):
                return expit(x)
            def sigmoiddif(x):
                return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
            return [sigmoid, sigmoiddif]

        elif acti == 'relu':
            def relu(x):
                return np.maximum(x, 0)
            def reludif(x):
                return np.where(x > 0, 1, 0)
            return [relu, reludif]

        elif acti == 'tanh':
            def tan(x):
                return np.tanh(x)
            def tandiff(x):
                return 1 - np.square(np.tanh(x))
            return [tan, tandiff]

        elif acti == 'softmax':
            def softmax(x):
                x = x - np.max(x)
                return np.exp(x) / np.sum(np.exp(x))

            def softmaxdif(output):
                return  output * (1 - output)
            return [softmax, softmaxdif]

        elif acti == "leakyrelu":
            def leakyrelu(x):
                return np.maximum(self.cvcoef * x, 0)
            def leakyreludif(x):
                return np.where(x > 0, self.cvcoef, 0)
            return [leakyrelu, leakyreludif]

        else:
            pass

    def forwardprop(self, input): #forward all the layers until output
        outlast = input
        activations = [input] #garder pour la backprop les variables
        zs = []
        for l in range(0, self.nblay):
            w = self.parameters["w" + str(l)]
            b = self.parameters["b" + str(l)]
            z = np.dot(w, outlast) + b
            if np.isinf(z).any():
                print("inf", np.isinf(z))
                print("________________________________________________________________")
                print("w", w)
                print("________________________________________________________________")
                print("out", outlast)
                print("________________________________________________________________")
                print("profuit", np.dot(w, outlast))
                print("_______________________________________________________________")
                print("b", b)
                print("________________________________________________________________")
                print("z", z)
                print("________________________________________________________________")
                raise Exception("something went wrong")
            a = self.parameters["fct" + str(l)](z)

            zs.append(z)
            activations.append(a)
            outlast = a

        return outlast, zs, activations #out last c'est la prediction et vieux c'est pour backprop

    def backprop(self, expected, zs, activations, nbinp=1):
        dw = []
        db = []
        delta = self.differrorfunc(activations[-1], expected, nbinp)

        dw.append(np.dot(delta, activations[-2].T))
        db.append(np.sum(delta, axis=1, keepdims=True))

        for l in range(self.nblay - 2, -1, -1):
            w = self.parameters["w" + str(l + 1)]
            dif = self.parameters["diff" + str(l)](zs[l])

            delta = np.dot(w.T, delta) * dif

            dwl = np.dot(delta, activations[l].T)
            dbl = np.sum(delta, axis=1, keepdims=True)

            dw.append(dwl)
            db.append(dbl)

        dw, db = [np.array(a) for a in dw[::-1]], [np.array(a) for a in db[::-1]]
        return dw, db

    def actualiseweights(self, dw, db):
        for l in range(0,self.nblay):
            self.parameters["w" + str(l)] -= self.cvcoef * dw[l] * (1/self.lenbatch)
            self.parameters["b" + str(l)] -= self.cvcoef * db[l] * (1/self.lenbatch)

    def trainsimple(self):
        for _ in range(self.iter):
            for p in range(len(self.pix)):
                forw = self.forwardprop(self.pix[p].reshape((784,1)))

                dw, db = self.backprop(self.vecteur(self.vales[p]), forw[1], forw[2], self.lenbatch)

                self.actualiseweights(dw, db)

    def choix(self, y):
        return np.argmax(y)

    def vecteur(self, val):
        return np.array([1 if i == val else 0 for i in range(10)]).reshape((10,1))

    def tauxerreur(self): #go in all the test and see accuracy
        nbbien = 0
        for image in range(len(self.qcmpix)):
            forw = self.forwardprop(self.qcmpix[image])

            observed = self.choix(forw[0])

            if observed == self.qcmval[image]:
                nbbien += 1

        return nbbien*100 / len(self.qcmpix)

    def prediction(self, image):
        forw = self.forwardprop(image)
        decision = self.choix(forw[0])
        return decision

val, pix, qcmval, qcmpix = takeinputs()

lay = [(784,"input"), (64,"sigmoid"), (10, "sigmoid")]

g = NN(pix, val, lay, "CEL", qcmpix, qcmval, iterations=1, batch=1)
#
# g.trainsimple()
#
# print(g.tauxerreur())
