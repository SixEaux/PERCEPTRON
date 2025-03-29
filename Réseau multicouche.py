#GENERAL
import numpy as np
from scipy.special import expit
from dataclasses import dataclass

#CONV
from scipy.signal import correlate2d
from skimage.measure import block_reduce

#PRINT
from tabulate import tabulate
from matplotlib import pyplot as plt

#ORGANIZACION
from Auxiliares import takeinputs, Draw


# PARA EL FUTURO:
# - Cambiar como entran los inputs en convolution para que sea en np.ndarray (es decir crear una array que sea un (28*28*60000) o hacer reshape a cada vez)
# - no tener que cambiar pix y pixconv a cada vez
# - Añadir que learning rate cambie con variacion de lost function


# np.seterr(all='raise')

@dataclass
class Parametros:
    pix : list or np.ndarray
    vales : np.ndarray
    qcmpix: list or np.ndarray
    qcmval: np.ndarray

    infolay: list
    infoconvlay: list

    iterations: int = 1
    coefcv: float = 0.01
    batch: int = 1
    errorfunc: str = "CEL"

    apprentissagedynamique: bool = False
    graph: bool = False
    color: bool = False

    #CNN
    kernel: int = 2
    kernelpool: int = 2
    padding: int = 1
    stride: int = 1
    poolstride: int = 2

    convrapide: bool = True


class CNN:
    def __init__(self, par = Parametros):

        self.iter = par.iterations  # nombre iteration entrainement
        self.nblay = len(par.infolay) # nombre de layers
        self.lenbatch = par.batch

        # INITIALISATION VARIABLES
        self.cvcoef = par.coefcv #learning rate

        # POUR CNN
        self.nbconv = len(par.infoconvlay) - 1
        self.lenkernel = par.kernel #lado filtro
        self.padding = par.padding #espacio con bordes
        self.stride = par.stride #de cuanto se mueve el filtro
        self.poolstride = par.poolstride
        self.lenkernelpool = par.kernelpool

        self.convolution = self.convolution2d if not par.convrapide else self.convolutionrapide

        self.convdims = [] #dimensiones salida convolution (dimconv, dimpool, nbfiltresentree, nbfiltressortie)

        d = 28
        for i in range(self.nbconv):
            dim = int(((d + 2 * self.padding - self.lenkernel) / self.stride) + 1) #dimension apres convolution
            dimpool = int(((dim - self.lenkernelpool) / self.poolstride) + 1)  #dimension apres pooling layer
            self.convdims.append((dim, dimpool))
            d = dim

        if self.nbconv > 0: #il faut ajuster input multilayer because it comes from convlayer
            par.infolay = [(self.convdims[-1][1], "input")] + par.infolay
        else:
            par.infolay = [(784, "input")] + par.infolay

        # INPUTS POUR ENTRAINEMENT
        self.pix = self.processdata(par.pix, par.color, False, self.nbconv>0) #pix de train
        self.vales = par.vales #val de train

        # BASE DE DONNÉES POUR LES TESTS
        self.qcmpix = self.processdata(par.qcmpix, par.color, True, self.nbconv>0)
        self.qcmval = par.qcmval

        self.parameters = self.params(par.infolay, par.infoconvlay) #creer les parametres dans un dico/ infolay doit avoir tout au debut la longueur de l'input
        self.dimweights = [(par.infolay[l][0], par.infolay[l-1][0]) for l in range(1, len(par.infolay))] #dimensiones pesos para backprop

        self.errorfunc = self.geterrorfunc(par.errorfunc) #choisir la fonction d'erreur

        self.fctconv = self.getfct("relu")

        self.aprentissagedynamique = par.apprentissagedynamique
        self.graph = par.graph

    def printbasesimple(self, base):
        print(tabulate(base.reshape((28, 28))))

    def printcouleur(self, base, titre):
        df2 = base.reshape((28,28))
        plt.imshow(df2, cmap='Greys', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.title(titre)
        plt.show()

    def converttogreyscale(self,rgbimage):
        return np.dot(rgbimage,[0.299, 0.587, 0.114])

    def processdata(self, pix, color, qcm, conv): #mettre les donnees sous la bonne forme
        if conv:
            if color:
                datamod = [self.converttogreyscale(a) / 255 for a in pix]
            else:
                if qcm:
                    datamod = [pix[:,a].reshape(28,28) for a in range(pix.shape[1])]
                else:
                    datamod = [pix[:,a].reshape(28,28) / 255 for a in range(pix.shape[1])]

        else: #si pas convolution direct avec numpy
            if color:
                pix = self.converttogreyscale(pix)

            if qcm:
                datamod = pix
            else:
                datamod = pix/255

        return datamod

    def params(self, infolay, infoconvlay): #infolay liste avec un tuple avec (nbneurons, fctactivation) / infoconvlay (nbfiltres, fct)
        param = {}

        for c in range(1, len(infoconvlay)):
            param["cl" + str(c-1)] = np.random.uniform(-1, 1, size=(self.lenkernel, self.lenkernel, infoconvlay[c-1][0], infoconvlay[c][0])) #(hauteur filtre, largeur filtre, nb canaux entree, nb canaux sortie)
            param["fctcl" + str(c-1)] = self.getfct(infoconvlay[c][1])
            self.convdims[c-1] = (self.convdims[c-1][0], self.convdims[c-1][1], infoconvlay[c-1][0], infoconvlay[c][0]) #antiguas dim con: añadir el numero filtros entrada y salida

        if self.nbconv > 0:
            infolay[0] = (infolay[0][0]*infolay[0][0]*self.convdims[self.nbconv-1][3], "input") #ajustar para que primer peso tenga buenas dim

        for l in range(1, len(infolay)):
            param["w" + str(l-1)] = np.random.uniform(-1, 1, (infolay[l][0], infolay[l-1][0])) #nbneurons * nbinput
            param["b" + str(l-1)] = np.random.rand(infolay[l][0], 1) - 0.5
            param["fct" + str(l-1)] = self.getfct(infolay[l][1])[0]
            param["diff" + str(l-1)] = self.getfct(infolay[l][1])[1]

        return param

    def geterrorfunc(self, errorfunc): #exp est un onehotvect
        if errorfunc == "eqm":
            def eqm(obs, exp, nbinput):
                return (np.sum((obs - exp) ** 2, axis=0))/ (2 * nbinput)
            def eqmdif(obs, expected, nbinput):
                return  (obs - expected)/nbinput
            return [eqm, eqmdif]

        elif errorfunc == "CEL":
            def CEL(obs, exp, nbinput):
                return -np.sum(exp * np.log(np.clip(obs, 1e-9, 1 - 1e-9)), axis=0) / nbinput
            def CELdif(obs, exp, nbinput):
                return (obs - exp) / nbinput
            return [CEL, CELdif]

        else:
            raise ValueError("errorfunc must be specified")

    def getfct(self, acti):
        if acti == 'sigmoid':
            def sigmoid(x):
                return expit(x)
            def sigmoiddif(x):
                return (expit(x)) * (1 - expit(x))
            return [sigmoid, sigmoiddif]

        elif acti == 'relu':
            def relu(x):
                return np.maximum(x, 0)
            def reludif(x):
                return np.where(x >= 0, 1, 0)
            return [relu, reludif]

        elif acti == 'tanh':
            def tan(x):
                return np.tanh(x)
            def tandiff(x):
                return 1 - np.square(np.tanh(x))
            return [tan, tandiff]

        elif acti == 'softmaxaprox':
            def softmaxaprox(x):
                x = x - np.max(x, axis=0, keepdims=True)
                return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

            def softmaxaproxdif(output):
                return output * (1 - output)

            return [softmaxaprox, softmaxaproxdif]

        elif acti == 'softmax':
            def softmax(x):
                x = x - np.max(x, axis=0, keepdims=True)
                return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

            def softmaxdif(output):
                n = output.shape[0]
                jacob = np.zeros((n, n))

                for i in range(n):
                    for j in range(n):
                        if i == j:
                            jacob[i, j] = output[i] * (1 - output[i])
                        else:
                            jacob[i, j] = -output[i] * output[j]

                return jacob

            return [softmax, softmaxdif]

        elif acti == "leakyrelu":
            def leakyrelu(x):
                return np.maximum(self.cvcoef * x, 0)

            def leakyreludif(x):
                return np.where(x > 0, self.cvcoef, 0)

            return [leakyrelu, leakyreludif]

        else:
            raise "You forgot to specify the activation function"

    def convolution2d(self, image, kernel, dimout=None): #dimout est calculer avant ou
        lenkernel = kernel.shape #(largeur,hauteur, canaux entree, canaux sortie)

        if dimout is None: #calcul dim sortie
            dimout = (int((image.shape[0] - lenkernel[0]) / self.stride) + 1, int((image.shape[1] - lenkernel[1]) / self.stride) + 1, lenkernel[3])

        output = np.zeros(dimout)

        for d in range(output.shape[2]): #parcours filtres

            for h in range(output.shape[0]): #parcours hauteur

                hdebut = h * self.stride
                hfin = hdebut + lenkernel[0]

                for l in range(output.shape[1]): #parcours largeur

                    ldebut = l * self.stride
                    lfin = ldebut + lenkernel[1]

                    output[h, l, d] += np.sum(image[hdebut:hfin, ldebut:lfin, :] * kernel[:,:, :, d]) #multiplie et somme sur canaux entree pour obtenir 2d

        return output

    def pooling(self, image): #para poder utilizar np.max hace falta un mask para guardar de donde viene el max / si average no hace falta

        dimout = (int((image.shape[0] - self.lenkernelpool) / self.poolstride) + 1,int((image.shape[1] - self.lenkernelpool) / self.poolstride) + 1, image.shape[2])
        output = np.zeros(dimout)

        for d in range(image.shape[2]):#parcours canaux

            for h in range(output.shape[0]): #parcours hauteur
                hdebut = h * self.poolstride
                hfin = hdebut + self.lenkernelpool

                for l in range(output.shape[1]): #parcours largeur
                    ldebut = l * self.poolstride
                    lfin = ldebut + self.lenkernelpool

                    output[h, l, d] += np.average(image[hdebut:hfin, ldebut:lfin, d]) #media feature map para poner en output

        return output

    def convolutionrapide(self, image, kernel, dimout=None):  # faire convolution avec librairie
        lenkernel = kernel.shape

        if dimout is None:
            dimout = (int((image.shape[0] - lenkernel[0]) / self.stride) + 1, int((image.shape[1] - lenkernel[1]) / self.stride) + 1, lenkernel[2])

        output = np.zeros(dimout)

        for d in range(lenkernel[2]):
            output[:, :, d] += correlate2d(image, kernel[:, :, d], "valid")

        return output

    def maxpoolingrapide(self, image):
        return block_reduce(image, (self.lenkernel, self.lenkernel), np.max)

    def flatening(self, image):
        return image.reshape((-1,1))

    def paddington(self, image, pad):
        return np.pad(image, (pad, pad))  # padding

    def forwardprop(self, input): #forward all the layers until output
        outlast = input

        if self.nbconv > 0:
            activationsconv = [input] #garder activees pour backprop des convolution
            activationslay = [] #garder activees pour la backprop les variables des layers
        else:
            activationsconv = []
            activationslay = [input]

        #ici garder seulement avant activation
        zslay = []
        zsconv = []

        for c in range(self.nbconv): #parcours layers convolution
            kernel = self.parameters["cl" + str(c)]
            conv = self.convolution(self.paddington(outlast, self.padding), kernel)
            pool = self.pooling(conv)

            if c == self.nbconv - 1: #si arrives a la fin flattening layer
                outlast = self.flatening(pool)
                zsconv.append(outlast)
            else: #sinon continue
                outlast = pool
                zsconv.append(outlast)

            outlast = self.fctconv[0](outlast)
            activationsconv.append(outlast)

        activationslay.append(activationsconv[-1])

        for l in range(self.nblay):
            w = self.parameters["w" + str(l)]
            b = self.parameters["b" + str(l)]
            z = np.dot(w, outlast) + b

            a = self.parameters["fct" + str(l)](z)

            zslay.append(z)
            activationslay.append(a)
            outlast = a

        return outlast, zslay, zsconv, activationslay, activationsconv #out last c'est la prediction et vieux c'est pour backprop

    def backpool(self, dapres, dimsortie): #pooling pero al reves, recuperar algo de mismas dim que entrada en pooling
        s = dapres.shape
        out = np.zeros(dimsortie)

        long = (self.lenkernelpool * self.lenkernelpool)

        for d in range(dimsortie[2]): #parcours canaux

            for h in range(s[0]): #parcours hauteur

                hdebut = h * self.poolstride
                hfin = hdebut + self.lenkernelpool

                for l in range(s[1]): #parcours largeur

                    ldebut = l * self.poolstride
                    lfin = ldebut + self.lenkernelpool

                    out[hdebut:hfin, ldebut:lfin, d] += np.full((self.lenkernelpool, self.lenkernelpool) , dapres[h,l,d] / long) #en toda la region ponemos la media
        return out

    def backprop(self, expected, zslay, zsconv, activationslay, activationsconv, nbinp):
        C = self.errorfunc[0](activationslay[-1], expected, nbinp) #Calcular error

        #crear los outputs
        dw = [np.zeros(self.dimweights[i]) for i in range(self.nblay)]
        db = [np.zeros((self.dimweights[i][0], 1)) for i in range(self.nblay)]
        dc = [np.zeros((self.lenkernel, self.lenkernel, self.convdims[i][2], self.convdims[i][3])) for i in range(self.nbconv)]

        delta = self.errorfunc[1](activationslay[-1], expected, nbinp) #error output layer

        dw[-1] += np.dot(delta, activationslay[-2].T) #dC/dpesos antes de salida
        db[-1] += np.sum(delta, axis=1, keepdims=True) #dC/dbias antes de salida

        for l in range(self.nblay - 2, -1, -1): #parcours layers à l'envers

            w = self.parameters["w" + str(l + 1)]
            dif = self.parameters["diff" + str(l)](zslay[l])

            delta = np.dot(w.T, delta) * dif #update error con error siguiente layer

            dwl = np.dot(delta, activationslay[l].T)
            dbl = np.sum(delta, axis=1, keepdims=True)

            dw[l] += dwl
            db[l] += dbl

        if self.nbconv>0:
            # Calcular ultimo delta para el conv layer
            ultimoweight = self.parameters["w0"]
            ultimadif = self.fctconv[1](zsconv[-1])

            s = self.convdims[-1] #dimensiones ultimo conv

            delta = (np.dot(ultimoweight.T, delta) * ultimadif).reshape(s[1],s[1], s[3]) #calcular ultimo error de nn

            delta = self.backpool(delta, (s[0], s[0], s[3])).reshape(s[0], s[0], s[2], s[3]) #recuperar misma talla que input de pooling

            dc[-1] += self.convolution(activationsconv[self.nbconv-1], delta).reshape(self.lenkernel, self.lenkernel, s[2], s[3])

            for c in range(self.nbconv - 2, -1, -1):
                filtre = self.parameters["cl" + str(c)]
                diff = self.fctconv[1](zsconv[c])

                delta = self.convolution(delta, filtre) * diff

                dLdf = self.convolution(activationsconv[c-1], delta)

                dc[c] += dLdf

        return dw, db, C, dc

    def actualiseweights(self, dw, db, nbinput, dc=None):
        for l in range(0,self.nblay):
            self.parameters["w" + str(l)] -= self.cvcoef * dw[l] * (1/nbinput)
            self.parameters["b" + str(l)] -= self.cvcoef * db[l] * (1/nbinput)

        for c in range(self.nbconv):
            self.parameters["cl" + str(c)] -= self.cvcoef * dc[c] * (1/nbinput)

        return

    def choix(self, y):
        return np.argmax(y,axis=0) #, keepdims=True

    def vecteur(self, val):
        if self.lenbatch == 1:
            newval = [val]
        else:
            newval = val
        return np.eye(10)[newval].T

    def train(self):
        if self.lenbatch > 1:
            self.trainbatch()
        elif self.lenbatch == 1:
            self.trainsimple()
        return

    def trainsimple(self):
        if self.nbconv == 0:
            C = []
            for _ in range(self.iter):
                L = []
                for p in range(self.pix.shape[1]):
                    forw = self.forwardprop(self.pix[:,p].reshape(-1,1))

                    dw, db, loss, dc = self.backprop(self.vecteur(self.vales[p]), forw[1], forw[2], forw[3], forw[4], 1)

                    self.actualiseweights(dw, db, 1, dc)

                    L.append(loss)

                C.append(np.average(L))

            if self.graph:
                plt.plot([i for i in range(self.iter)], C)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Fonction de Cout')
                plt.show()

        else:
            for _ in range(self.iter):
                for p in range(len(self.pix)):
                    forw = self.forwardprop(self.pix[p].reshape(28,28,-1))

                    dw, db, loss, dc = self.backprop(self.vecteur(self.vales[p]), forw[1], forw[2], forw[3], forw[4], 1)

                    self.actualiseweights(dw, db, 1, dc)

                    if p % 1000 == 0:
                        print("Percentage: " + str(p*100/len(self.pix)))

        return

    def trainbatch(self):
        if self.nbconv == 0:
            for _ in range(self.iter):
                nbbatch = self.pix.shape[1] // self.lenbatch
                for bat in range(nbbatch):
                    matrice = self.pix[:, bat*self.lenbatch:(bat+1)*self.lenbatch].reshape(-1, self.lenbatch)

                    forw = self.forwardprop(matrice)

                    dw, db, loss, dc = self.backprop(self.vecteur(self.vales[bat*self.lenbatch:(bat+1)*self.lenbatch]), forw[1], forw[2], forw[3], forw[4], self.lenbatch)

                    self.actualiseweights(dw, db, self.lenbatch)

        else:
            print("EN TRAVAUX")

        return

    def tauxlent(self): #go in all the test and see accuracy
        if self.nbconv == 0:
            nbbien = 0
            for image in range(self.qcmpix.shape[1]):
                forw = self.forwardprop(self.qcmpix[:, image].reshape(-1,1))

                observed = self.choix(forw[0])

                if observed == self.qcmval[image]:
                    nbbien += 1
                else:
                    if self.aprentissagedynamique:

                        dw, db, _, dc = self.backprop(self.vecteur(self.vales[image]), forw[1], forw[2], forw[3], forw[4], 1)


                        self.actualiseweights(dw, db, 1, dc)

            return nbbien*100 / self.qcmpix.shape[1]

        else:
            nbbien = 0
            for image in range(len(self.qcmpix)):
                forw = self.forwardprop(self.qcmpix[image].reshape(28,28,-1))

                observed = self.choix(forw[0])

                if observed == self.qcmval[image]:
                    nbbien += 1

            return nbbien * 100 / len(self.qcmpix)

    def tauxrapide(self):
        if self.nbconv == 0:
            forw = self.forwardprop(self.qcmpix.reshape(784,-1))

            observed = self.choix(forw[0])

            difference = observed - self.qcmval

            nbbien = np.count_nonzero(difference==0)

            return nbbien*100 / self.qcmpix.shape[1]
        else:
            print("EN TRAVAUX")
            return

    def prediction(self, image):
        self.printcouleur(image, "")
        forw = self.forwardprop(image)
        decision = self.choix(forw[0])
        print(f"Je crois bien que cela est un {decision}")
        return

    def TryToDraw(self):
        cnv = Draw()

        px = cnv.pixels

        self.prediction(px)

    def graphisme(self):
        fct = []
        for _ in range(self.iter):
            self.trainsimple()
            a = self.tauxlent()
            fct.append(a)
        plt.plot([i for i in range(len(fct))], fct)
        plt.xlabel('Iteration')
        plt.ylabel('Taux erreur')
        plt.title('Fonction de Erreur')
        plt.show()


val, pix, qcmval, qcmpix, pixelsconv, qcmpixconv = takeinputs()

convlay = [(1, "input"), (3, "relu")]

lay = [(64, "sigmoid"), (10, "softmax")]

parametros = Parametros(pix=pix, vales=val, qcmpix=qcmpix, qcmval=qcmval, infolay=lay, infoconvlay=convlay, padding=0, convrapide=False)

g = CNN(parametros)

# g.train()

print(g.tauxlent())


