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
        petitqcmval = np.array(qcmval[0:5000])

    with open('testpix', 'rb') as f:
        qcmpix = np.array(pickle.load(f))
        petitqcmpix = np.array(qcmpix[0:5000])

    return valeurs, pixels


class NN:
    def __init__(self, pix, vales, infolay, errorfunc, *, coefcv=0.1, iterations=10):
        self.iter = iterations  # nombre iteration entrainement
        self.nblay = len(infolay) # nombre de layers

        # INITIALISATION VARIABLES
        self.cvcoef = coefcv

        # INPUTS POUR ENTRAINEMENT
        self.pix = pix/255 #pix de train
        self.vales = vales #val de train

        self.parameters = self.params(infolay) #creer les parametres dans un dico/ infolay doit avoir tout au debut la longueur de l'input

        self.errorfunc = self.geterrorfunc(errorfunc)[0] #choisir la fonction d'erreur
        self.differrorfunc = self.geterrorfunc(errorfunc)[1]


    def printbasesimple(self, base):
        print(tabulate(base.reshape((28, 28))))

    def geterrorfunc(self, errorfunc): #exp est un onehotvect
        if errorfunc == "eqm":
            return [lambda obs, exp, nbinput=1: (np.sum((obs - exp) ** 2, axis=1)) / (2 * nbinput), lambda obs, expected, nbinput=1: (obs - expected)/nbinput]
        elif errorfunc == "CCC":
            return [lambda obs, expected: -np.sum(expected * np.log(np.clip(obs, 1e-7, 1 - 1e-7)), axis=1), None] #si le exp c'est un one hot ver¡cteurs
            # si place bon output: return lambda obs, exp: -np.log(np.clip(obs, 1e-7, 1 - 1e-7)[exp, 1])
            # il manque la diff

    def getfct(self, acti):
        if acti == 'sigmoid':
            return [lambda x: 1 / (1 + np.exp(-x)), lambda x: np.exp(-x) / (1 + np.square(np.exp(-x)))]

        elif acti == 'relu':
            return [lambda x: np.where(x > 0, x, 0), lambda x: np.where(x > 0, x, 0)]

        elif acti == 'tanh':
            return [lambda x: np.tanh(x), lambda x: 1 - np.square(np.tanh(x))]

        elif acti == 'softmax':
            return [lambda input: np.exp(input) / np.sum(np.exp(input), axis=0), None]

        else:
            pass

    def params(self, lst): #lst liste avec un tuple avec (nbneurons, fctactivation)
        param = {}

        for l in range(1, len(lst)):
            param["w" + str(l)] = np.random.uniform(-1,1,(lst[l-1][0], lst[l][0]))
            param["b" + str(l)] = np.random.uniform(-1,1,(lst[l][0], 1))
            param["fct" + str(l)] = self.getfct(lst[l][1])[0]
            param["diff" + str(l)] = self.getfct(lst[l][1])[1]
        return param


    def forwardprop(self, input): #forward all the layers until output
        outlast = input
        vieux = [(input,0,0,0)] #garder pour la backprop les variables
        for l in range(1, self.nblay-1):
            w = self.parameters["w" + str(l)]
            b = self.parameters["b" + str(l)]
            z = np.dot(w.T, outlast) + b
            a = self.parameters["fct" + str(l)](z)
            vieux.append((a,w,b,z))
            outlast = a

        w = self.parameters["w" + str(self.nblay-1)]
        b = self.parameters["b" + str(self.nblay-1)]
        z = np.dot(w.T, outlast) + b
        outlast = self.parameters["fct" + str(self.nblay-1)](z)
        vieux.append((outlast, w, b, z))

        return outlast, vieux #out last c'est la prediction et vieux c'est pour backprop

    def backprop(self, observed, expected, vieux, nbinp=1): # observed y expected vectores de nboutputs * 1
        C = self.errorfunc(observed,expected, nbinp)
        dCda = self.differrorfunc(observed,expected, nbinp)
        print(dCda)

        activ_prec, w, b, zO = vieux[-1]
        dactivzO = self.parameters["diff" + str(self.nblay-1)](zO)

        dO = dCda * dactivzO

        self.parameters["w" + str(self.nblay - 1)] -= self.cvcoef * np.dot(activ_prec, dO.T)
        self.parameters["b" + str(self.nblay - 1)] -= self.cvcoef * np.sum(dO, axis=1, keepdims=True)

        dL = dO
        for i in range(self.nblay - 2, 0, -1):
            activ_prec, w, b, z = vieux[i]
            dactiv = self.parameters["diff" + str(i)](z)
            dL = np.dot(self.parameters["w" + str(i + 1)], dL) * dactiv

            self.parameters["w" + str(i)] -= self.cvcoef * np.dot(dL, activ_prec.T)
            self.parameters["b" + str(i)] -= self.cvcoef * np.sum(dL, axis=1, keepdims=True)
            print(self.parameters["w" + str(i)])

    def newbackprop(self, observed, expected, vieux, nbinp=1):

        activ, _, _, z = vieux[-1]
        delta = self.differrorfunc(observed, expected, nbinp) * self.parameters["diff" + str(self.nblay-1)](z)
        print(delta.shape)
        print(vieux[-2][0].T.shape)


        self.parameters["w" + str(self.nblay - 1)] -= self.cvcoef * np.dot(vieux[-2][0], delta.T)
        self.parameters["b" + str(self.nblay - 1)] -= self.cvcoef * delta

        for l in range(self.nblay-2, 0, -1):
            activ, w, _, z = vieux[l]
            diff = self.parameters["diff" + str(l)](z)
            delta = np.dot(w.T, delta) * diff

            self.parameters["w" + str(l)] -= self.cvcoef * np.dot(delta, vieux[-2][0].transpose())
            self.parameters["b" + str(l)] -= self.cvcoef * delta



    def trainsimple(self):
        pass


    def tauxerreur(self): #go in all the test and see accuracy
        pass


def vecteur(val):
    return np.array([1 if i ==val-1 else 0 for i in range(10)])


val, pix = takeinputs()

lay = [(784,"input"), (4, "relu"), (10, "sigmoid")]

g = NN(pix, val, lay, "eqm")


l = g.forwardprop((pix[10].reshape(784,1))/255)

g.newbackprop(l[0], vecteur(val[10]).reshape((10,1)), l[1])

