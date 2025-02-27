import numpy as np
import pickle

from tabulate import tabulate

# Plus tard:
# do method with batchs to do it directly with 32 for example
# le input est un batch en forme de matrice avec 784 lignes et 32 colonnes
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
        petitqcmval = np.array(qcmval[0:50])

    with open('testpix', 'rb') as f:
        qcmpix = np.array(pickle.load(f))
        petitqcmpix = np.array(qcmpix[0:50])

    return valeurs, pixels, petitqcmval, petitqcmpix


class NN:
    def __init__(self, pix, vales, infolay, errorfunc, qcmpix, qcmval, *, coefcv=0.1, iterations=1):
        self.iter = iterations  # nombre iteration entrainement
        self.nblay = len(infolay)-1 # nombre de layers

        # INITIALISATION VARIABLES
        self.cvcoef = coefcv

        # INPUTS POUR ENTRAINEMENT
        self.pix = pix/255 #pix de train
        self.vales = vales #val de train

        self.qcmpix = qcmpix/255
        self.qcmval = qcmval

        self.parameters = self.params(infolay) #creer les parametres dans un dico/ infolay doit avoir tout au debut la longueur de l'input

        self.errorfunc = self.geterrorfunc(errorfunc)[0] #choisir la fonction d'erreur
        self.differrorfunc = self.geterrorfunc(errorfunc)[1]


    def printbasesimple(self, base):
        print(tabulate(base.reshape((28, 28))))

    def geterrorfunc(self, errorfunc): #exp est un onehotvect
        if errorfunc == "eqm":
            return [lambda obs, exp, nbinput=1: (np.sum((obs - exp) ** 2, axis=1)) / (2 * nbinput), lambda obs, expected, nbinput=1: (obs - expected)/nbinput]

    def getfct(self, acti):
        if acti == 'sigmoid':
            return [lambda x: 1 / (1 + np.exp(-x)), lambda x: - np.exp(-x) / np.square(1 + np.exp(-x))]

        elif acti == 'relu':
            return [lambda x: np.where(x > 0, x, 0), lambda x: np.where(x > 0, 1, 0)]

        elif acti == 'tanh':
            return [lambda x: np.tanh(x), lambda x: 1 - np.square(np.tanh(x))]

        elif acti == 'softmax':
            return [lambda input: np.exp(input) / np.sum(np.exp(input), axis=0), None]

        else:
            pass

    def params(self, lst): #lst liste avec un tuple avec (nbneurons, fctactivation)
        param = {}

        for l in range(1, len(lst)):
            param["w" + str(l-1)] = np.random.uniform(-1, 1, (lst[l - 1][0], lst[l][0]))
            param["b" + str(l-1)] = np.random.uniform(-1, 1, (lst[l][0], 1))
            param["fct" + str(l-1)] = self.getfct(lst[l][1])[0]
            param["diff" + str(l-1)] = self.getfct(lst[l][1])[1]

        return param


    def forwardprop(self, input): #forward all the layers until output
        outlast = input
        activations = [input] #garder pour la backprop les variables
        zs = []
        for l in range(0, self.nblay):
            w = self.parameters["w" + str(l)]
            b = self.parameters["b" + str(l)]
            z = np.dot(w.T, outlast) + b
            a = self.parameters["fct" + str(l)](z)

            zs.append(z)
            activations.append(a)

            outlast = a

        return outlast, zs, activations #out last c'est la prediction et vieux c'est pour backprop

    def backprop(self, expected, zs, activations, input, nbinp=1): # observed y expected vectores de nboutputs * 1
        dw = []
        db = []
        delta = self.differrorfunc(activations[-1], expected, nbinp)

        dw.append(np.dot(activations[-2], delta.T))
        db.append(np.sum(delta, axis=1, keepdims=True))

        for l in range(self.nblay-2, 0, -1):

            w = self.parameters["w" + str(l+1)]

            dif = self.parameters["diff" + str(l)](zs[l])

            delta = np.dot(w, delta) * dif

            dwl = np.dot(activations[l-1], delta.T)
            # print(np.dot(activations[l-1], delta.T).shape)
            dbl = np.sum(delta, axis=1, keepdims=True)

            dw.append(dwl)
            db.append(dbl)

        w = self.parameters["w1"]
        dif = self.parameters["diff0"](zs[0])
        delta = np.dot(w, delta) * dif

        dw.append(np.dot(input, delta.T))
        db.append(np.sum(delta, axis=1, keepdims=True))

        return dw[::-1], db[::-1]

    def actualiseweights(self, dw, db):

        for l in range(1,self.nblay):
            self.parameters["w" + str(l)] -= dw[l]
            self.parameters["b" + str(l)] -= db[l]


    def trainsimple(self):
        for round in range(self.iter):
            for p in range(len(self.pix)):
                foto = pix[p].reshape(784, 1)
                forw = self.forwardprop(foto)

                dw, db = self.backprop(forw[0], self.vecteur(self.vales[p]), forw[1], foto)

                # print([a.shape for a in dw])
                # print([a.shape for a in db])

                self.actualiseweights(dw, db)

    def vraievaleur(self, y):
        return np.argmax(y)

    def tauxerreur(self): #go in all the test and see accuracy
        nbbien = 0
        for image in range(len(self.qcmpix)):
            foto = self.qcmpix[image].reshape(784, 1)
            forw = self.forwardprop(foto)

            observed = self.vraievaleur(forw[0])

            if observed == self.qcmval[image]:
                nbbien += 1

        return nbbien


    def vecteur(self, val):
        return np.array([1 if i == val - 1 else 0 for i in range(10)]).reshape((10,1))


val, pix, qcmval, qcmpix = takeinputs()

lay = [(784,"input"), (64,"relu"), (10, "relu")]

g = NN(pix, val, lay, "eqm", qcmpix, qcmval)

g.trainsimple()

print(g.tauxerreur())


