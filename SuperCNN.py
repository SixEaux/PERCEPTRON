#GENERAL
import time

from statistics import mean

import numpy as np
from scipy.special import expit

from dataclasses import dataclass

#CONV
from scipy.signal import correlate2d
from scipy.signal import convolve2d

from skimage.measure import block_reduce

#PRINT
from tabulate import tabulate
from matplotlib import pyplot as plt

#ORGANIZACION
from Auxiliares import takeinputs, Draw

from functools import wraps
from collections import defaultdict
import atexit

# PROBLEMAS:
# - convolutionnp
# - backpoolnp sans boucle
# - back convolution sans boucle

# PARA EL FUTURO:
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

    iterations: int = 10
    coefcv: float = 0.001
    coefcvadaptatif: bool = False
    batch: int = 1
    errorfunc: str = "CEL"

    apprentissagedynamique: bool = False
    graph: bool = False
    color: bool = False

    #CNN
    kernel: int = 3
    kernelpool: int = 2
    padding: int = 0
    stride: int = 1 #pour l'instant garder en 1

    poolnp: bool = True
    convnp: bool = True
    backconvnp: bool = True

execution_times = defaultdict(list)

def timed(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        execution_times[method.__name__].append(end - start)
        return result
    return wrapper

def maxdico(d):
    m = (None, 0)
    for i in d.keys():
        t = mean(d[i])
        if t > m[1]:
            m = (i, t)
    return m

@atexit.register
def print_avg_times():
    print("\n--- Temps moyens d'exécution ---")
    for name, times in execution_times.items():
        avg = mean(times)
        print("__________________________________________________________________________________")
        print(f"{name}: {avg} s    (appelée {len(times)} fois)")

    print("______________________________________________________________________________________")
    a = maxdico(execution_times)
    print(f"Le maximum de temps est de {a[1]} par {a[0]}")

class CNN:
    def __init__(self, par = Parametros):

        self.iter = par.iterations  # nombre iteration entrainement
        self.nblay = len(par.infolay) # nombre de layers
        self.lenbatch = par.batch

        # INITIALISATION VARIABLES
        self.cvcoef = par.coefcv #learning rate
        self.lradapt = par.coefcvadaptatif #adapte coefcv en fonction de la fonction de cout

        # POUR CNN
        self.nbconv = len(par.infoconvlay) - 1
        self.lenkernel = par.kernel #lado filtro
        self.padding = par.padding #espacio con bordes
        self.stride = par.stride #de cuanto se mueve el filtro
        self.poolstride = par.kernelpool
        self.lenkernelpool = par.kernelpool

        self.convdims = [] # (dimconv, dimpool, nbfiltresentree, nbfiltressortie)

        self.pooling = self.poolingnp if par.poolnp else self.poolingskim
        self.convolution = self.convolutionnp if par.convnp else self.convolutionscp
        self.backconvolution = self.backconvolutionnpbest if par.backconvnp else self.backconvolutionscp

        d = 28
        for i in range(self.nbconv):
            dim = int(((d + 2 * self.padding - self.lenkernel) / self.stride) + 1) #dimension apres convolution
            if par.infoconvlay[i+1][2]:
                dimpool = int(((dim - self.lenkernelpool) / self.poolstride) + 1)  #dimension apres pooling layer si dimensions paires
            else:
                dimpool = dim

            self.convdims.append((dim, dimpool))
            d = dimpool

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

    @timed
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
            param["cl" + str(c-1)] = np.random.uniform(-1, 1, size= (infoconvlay[c][0], infoconvlay[c-1][0], self.lenkernel, self.lenkernel)) # kernel: (nb canaux sortie, nb canaux entree, hauteur filtre, largeur filtre)
            param["cb" + str(c-1)] = np.zeros((infoconvlay[c][0], self.convdims[c-1][0], self.convdims[c-1][0])) # biais: canaux sortie, hauteur output, largeur output
            param["fctcl" + str(c-1)] = self.getfct(infoconvlay[c][1])
            param["pool" + str(c-1)] = infoconvlay[c][2]
            self.convdims[c-1] = (self.convdims[c-1][0], self.convdims[c-1][1], infoconvlay[c-1][0], infoconvlay[c][0]) #añadir el numero filtros entrada y salida

        if self.nbconv > 0:
            infolay[0] = (infolay[0][0]*infolay[0][0]*self.convdims[self.nbconv-1][3], "input") #ajustar para que primer peso tenga buenas dim

        for l in range(1, len(infolay)):
            param["w" + str(l-1)] = np.random.uniform(-1, 1, (infolay[l][0], infolay[l-1][0])) #nbneurons * nbinput
            param["b" + str(l-1)] = np.zeros((infolay[l][0], 1))
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

    @timed
    def convolutionnp(self, image, kernel, *, mode="valid", reverse=False):  # 2 casos dependiendo de shape kernel y imagen
        lenkernel = kernel.shape  # Csortie, Centree, H,L

        if mode == "full":
            newimage = self.paddington(image, lenkernel[2]-1, lenkernel[3]-1)
        elif mode == "valid":
            newimage = image
        else:
            raise ValueError("mode must be 'full' or 'valid'")

        if len(lenkernel) == 4:

            mapa = np.lib.stride_tricks.sliding_window_view(newimage, (lenkernel[2], lenkernel[3]), axis=(1, 2))

            if not reverse: #forward prop
                output = np.tensordot(mapa, kernel, axes=([0, 3, 4], [1, 2, 3])).transpose((2, 0, 1)) #essayer d'enlever le transpose pour opti
            else:
                output = np.tensordot(mapa, kernel, axes=([0, 3, 4], [0, 2, 3])).transpose((2, 0, 1))

        elif len(lenkernel) == 3:

            mapa = np.lib.stride_tricks.sliding_window_view(newimage, (lenkernel[1], lenkernel[2]), axis=(1, 2))

            output = np.tensordot(mapa, kernel, axes=([3, 4], [1, 2])).transpose(3,0,1,2)

        else:
            raise "Problem with the shapes they are not good"

        return output

    def convolutionscp(self, image, kernel, *, dimout=None, mode=None):

        lenkernel = kernel.shape  # (sortie, entree, hauteur,largeur)

        if dimout is None:  # calcul dim sortie
            dimout = (lenkernel[0], int((image.shape[1] - self.lenkernel) / self.stride) + 1, int((image.shape[2] - self.lenkernel) / self.stride) + 1)

        output = np.zeros(dimout)

        for d in range(dimout[0]):
            for ce in range(image.shape[0]):
                output[d] += correlate2d(image[ce], kernel[d,ce], mode="valid")[::self.stride,::self.stride]

        return output

    def poolingskim(self, image):
        d, h, l = image.shape

        if h % self.lenkernelpool == 0:
            newdims = (d, int((h - self.lenkernelpool) / self.lenkernelpool) + 1, int((l - self.lenkernelpool) / self.lenkernelpool) + 1)

            output = np.zeros(newdims)

            for c in range(newdims[0]):
                output[c] += block_reduce(image[c], (self.lenkernelpool, self.lenkernelpool), func=np.mean)

        else:
            newdims = (d, int((h - self.lenkernelpool) / self.lenkernelpool) + 1, int((l - self.lenkernelpool) / self.lenkernelpool) + 1)

            output = np.zeros(newdims)

            for c in range(d):
                output[c] = block_reduce(image[c], (self.lenkernelpool, self.lenkernelpool), func=np.mean)[:newdims[1], :newdims[2]]

        return output

    @timed
    def poolingnp(self, image):
        division = np.lib.stride_tricks.sliding_window_view(image, (self.lenkernelpool, self.lenkernelpool), axis=(1, 2))[:, ::self.lenkernelpool, ::self.lenkernelpool]
        return np.average(division, axis=(3, 4))

    @timed
    def flatening(self, image):
        return image.reshape((-1,1))

    @timed
    def paddington(self, image, padavant, padapres): #padavant ce qu'on ajoute a la ligne et l'autre est evident
        return np.pad(image, ((0,0), (padavant, padapres), (padavant, padapres))) # padding

    def forwardprop(self, input): #forward all the layers until output
        outlast = input

        activationsconv = [input] #garder activees pour backprop des convolution
        activationslay = [] #garder activees pour la backprop les variables des layers

        #ici garder seulement avant activation
        zslay = []
        zsconv = []

        for c in range(self.nbconv): #parcours layers convolution
            kernel = self.parameters["cl" + str(c)]
            biais = self.parameters["cb" + str(c)]

            if self.padding > 0:
                paded = self.paddington(outlast, self.padding, self.padding) #c'est surement mal
            else:
                paded = outlast

            conv = self.convolution(paded, kernel) + biais

            if self.parameters["pool" + str(c)]:
                pool = self.pooling(conv)
            else:
                pool = conv

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

    def backpoolnp(self, dapres, dimsortie):
        moyenne = dapres / (self.lenkernelpool * self.lenkernelpool)

        if dimsortie[1] % self.lenkernelpool == 0: #si pile
            output = np.zeros(dimsortie)

            for d in range(dapres.shape[0]):
                output[d] = np.repeat(np.repeat(moyenne[d], self.lenkernelpool, axis=0), self.lenkernelpool, axis=1) #on recree un kernel avec les dimensions

        else:
            c, h, l = dimsortie

            dif = h % self.lenkernelpool, l % self.lenkernelpool #si pas pile

            newh, newl = h - (dif[0]), l - (dif[1])

            output = np.zeros(dimsortie)

            for d in range(dapres.shape[0]):
                output[d, :newh, :newl] = np.repeat(np.repeat(moyenne[d], self.lenkernelpool, axis=0), self.lenkernelpool, axis=1)

        return output

    def backconvolutionscp(self, activation, dapres, filtre):
        gradc = np.zeros(filtre.shape)

        newdelta = np.zeros(activation.shape)

        for d in range(gradc.shape[0]):
            for c in range(activation.shape[0]):
                gradc[d, c] += correlate2d(activation[c, ::self.stride, ::self.stride], dapres[d], mode="valid")
                newdelta[c] += convolve2d(dapres[d], filtre[d,c], mode="full")

        return gradc, newdelta

    @timed
    def backconvolutionnpbest(self, activation, dapres, filtre):
        #pad image pour delta
        #convolution comme avant mais en inversant kernel

        gradc = self.convolution(activation, dapres)

        newdelta = self.convolution(dapres, np.flip(filtre, axis=(2,3)), mode="full", reverse=True)

        return gradc, newdelta

    def backprop(self, expected, zslay, zsconv, activationslay, activationsconv, nbinp):
        C = self.errorfunc[0](activationslay[-1], expected, nbinp) #Calcular error

        #crear los outputs
        dw = [np.zeros(self.dimweights[i]) for i in range(self.nblay)]
        db = [np.zeros((self.dimweights[i][0], 1)) for i in range(self.nblay)]
        dc = [np.zeros((self.convdims[i][3], self.convdims[i][2], self.lenkernel, self.lenkernel)) for i in range(self.nbconv)]
        dcb = [np.zeros(self.parameters["cb" + str(i)].shape) for i in range(self.nbconv)]

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

            delta = (np.dot(ultimoweight.T, delta) * ultimadif).reshape(s[3], s[1],s[1]) #calcular ultimo error de nn

            if self.parameters["pool" + str(self.nbconv-1)]:
                delta = self.backpoolnp(delta, (s[3], s[0], s[0])) #recuperar misma talla que input de pooling

            gradc, newdelta = self.backconvolution(activationsconv[self.nbconv - 1], delta, self.parameters["cl" + str(self.nbconv - 1)])

            dc[-1] += gradc
            dcb[-1] += delta

            for c in range(self.nbconv - 2, -1, -1):
                diff = self.fctconv[1](zsconv[c])

                delta = newdelta * diff

                s = self.convdims[c]

                #backpool
                if self.parameters["pool" + str(c)]:
                    delta = self.backpoolnp(delta, (s[3], s[0], s[0]))  # recuperar misma talla que input de pooling

                gradc, newdelta = self.backconvolution(activationsconv[c], delta, self.parameters["cl" + str(c)])

                dc[c] += gradc
                dcb[c] += delta

        return dw, db, C, dc, dcb

    @timed
    def actualiseweights(self, dw, db, nbinput, dc=None, dcb=None):
        for w in range(max(self.nblay,self.nbconv)):
            if w < self.nblay:
                self.parameters["w" + str(w)] -= self.cvcoef * dw[w] * (1 / nbinput)
                self.parameters["b" + str(w)] -= self.cvcoef * db[w] * (1 / nbinput)
            if w < self.nbconv:
                self.parameters["cl" + str(w)] -= self.cvcoef * dc[w] * (1 / nbinput)
                self.parameters["cb" + str(w)] -= self.cvcoef * dcb[w] * (1 / nbinput)
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

                    dw, db, loss, dc, dcb = self.backprop(self.vecteur(self.vales[p]), forw[1], forw[2], forw[3], forw[4], 1)

                    self.actualiseweights(dw, db, 1, dc, dcb)

                    L.append(loss)

                C.append(np.average(L))

            if self.graph:
                plt.plot([i for i in range(self.iter)], C)
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Fonction de Cout')
                plt.show()

        else:
            for i in range(self.iter):
                for p in range(len(self.pix)):
                    forw = self.forwardprop(self.pix[p].reshape(1,28,28)) #canaux, h,l

                    dw, db, loss, dc, dcb = self.backprop(self.vecteur(self.vales[p]), forw[1], forw[2], forw[3], forw[4], 1)

                    self.actualiseweights(dw, db, 1, dc, dcb)

                    if (p + self.iter*i) % 10000 == 0:
                        print("Percentage: " + str(np.round((p+i*len(self.pix))*100/(len(self.pix)*self.iter))))

        return

    def trainbatch(self):
        if self.nbconv == 0:
            for _ in range(self.iter):
                nbbatch = self.pix.shape[1] // self.lenbatch
                for bat in range(nbbatch):
                    matrice = self.pix[:, bat*self.lenbatch:(bat+1)*self.lenbatch].reshape(-1, self.lenbatch)

                    forw = self.forwardprop(matrice)

                    dw, db, loss, dc, dcb = self.backprop(self.vecteur(self.vales[bat*self.lenbatch:(bat+1)*self.lenbatch]), forw[1], forw[2], forw[3], forw[4], self.lenbatch)

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

                        dw, db, _, dc, dcb = self.backprop(self.vecteur(self.vales[image]), forw[1], forw[2], forw[3], forw[4], 1)


                        self.actualiseweights(dw, db, 1, dc, dcb)

            return nbbien*100 / self.qcmpix.shape[1]

        else:
            nbbien = 0
            for image in range(len(self.qcmpix)):
                forw = self.forwardprop(self.qcmpix[image].reshape(1,28,28))

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

convlay = [(1, "input"), (10, "relu", True)]
# convlay = [(1, "input"), (16, "relu", True), (32, "relu", True)]

lay = [(64, "sigmoid"), (10, "softmax")]

parametros = Parametros(pix=pix, vales=val, qcmpix=qcmpix, qcmval=qcmval, infolay=lay, infoconvlay=convlay, iterations=10)

g = CNN(parametros)

print("je commence a mentrainer")
t = time.time()

g.train()

print("jai fini en :", time.time()-t)

print("taux de reussite", g.tauxlent())


# forw = g.forwardprop(g.pix[10].reshape(1,28,28)) #canaux, h,l
#
# dw, db, loss, dc, dcb = g.backprop(g.vecteur(g.vales[10]), forw[1], forw[2], forw[3], forw[4], 1)
#
# g.actualiseweights(dw, db, 1, dc, dcb)

