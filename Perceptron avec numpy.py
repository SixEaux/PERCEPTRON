import numpy as np
import pickle
import random
from tabulate import tabulate
import matplotlib.pyplot as plt

with open('Datas/valeursentraine', 'rb') as f:
    valeurs = np.array(pickle.load(f))
    vali = np.array(valeurs[:10000])

with open('Datas/pixelsentraine', 'rb') as f:
    pixels = np.array(pickle.load(f))
    pixi = np.array(pixels[:10000])

with open('Datas/testval', 'rb') as f:
    qcmval = np.array(pickle.load(f))
    petitqcmval = np.array(qcmval[0:5000])

with open('Datas/testpix', 'rb') as f:
    qcmpix = np.array(pickle.load(f))
    petitqcmpix = np.array(qcmpix[0:5000])

class Perceptron:
    def __init__(self, pix, vales, *, nbneurones = 784, coefcv = 0.1, iterations=10, seuil = 0.0, normal = False, apprentissagedynamique = False,
                 bruitgaussien = False, pourcecarttype = 0.0,
                 bruitsurpix = False, nbpixelsbruit = 0, positionschoisies = False, saturation = True):

        self.iter = iterations #nombre iteration entrainement
        self.nb = nbneurones #nombre de neurones

        #INITIALISATION VARIABLES
        self.poids = np.ones(nbneurones + 1) #np.random.randn(nbneurones + 1) * 0.01 #avec le biai qui est la premiere valeur
        self.cvcoef = coefcv
        self.seuil = seuil
        self.biais = 1

        #INPUTS POUR ENTRAINEMENT
        self.pix = pix
        self.vales = vales


        #BRUIT GAUSSIEN
        self.normal = normal
        self.boolbruitgaus = bruitgaussien
        self.pix = self.normaliserbase(self.pix)
        self.ecarttype = pourcecarttype / 100

        #BRUIT PAR PIXELS
        self.boolbruitpix = bruitsurpix
        self.saturation = saturation #estce qu'on met les pixels noirs ou blancs
        self.choisies = positionschoisies
        self.nbbruit = nbpixelsbruit

        #APPRENTISSAGE DYNAMIQUE
        self.dynamique = apprentissagedynamique

    def normaliserbase(self, base):
        return base/255

    def bruitgaussien(self, image):
        bruit = np.random.normal(0, self.ecarttype, 784)
        imagebrouillee = np.clip(image + bruit, 0, 1) #je ne sais pas si nécessaire mais pour qu'il n'y ait pas de valeur au de la
        return imagebrouillee

    def bruitpixels(self, image, norm, saturation): #choix de x pixels pour les mettre a noir ou blanc
        sat = (1,255) if saturation else (0,0)
        if self.choisies:
            try:
                posbruits = list(map(int,list(input(f"Donnez moi une liste avec les positions que vous voulez modifier \n"
                                                    f" nombres entre 0 et 783 et separes par des virgules: ").split(","))))
                if len(posbruits) > self.nbbruit:
                    posbruits = posbruits[:self.nbbruit]
                print(posbruits)
            except:
                print("Ce n'est pas le format correct donc pas de bruit")
                return image
        else:
            posbruits = random.sample(range(0, 783), self.nbbruit)

        for el in posbruits:
            image[el] = sat[0] if norm else sat[1]

        return image

    def printbasesimple(self, base):
        print(tabulate(base[1:].reshape((28,28))))

    def printcouleur(self, base, titre):
        df2 = base[1:].reshape((28,28))
        plt.imshow(df2, cmap='Greys', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.title(titre)
        plt.show()

    def fctactivescalier(self, x):
        return 1 if x>self.seuil else 0

    def produit(self,tab1,tab2):
        return np.dot(tab1,tab2)

    def vraiinput(self, input): #ajouter biais a l'input
        return np.concatenate(([self.biais],input))

    def changerpoids(self, attendu, observe, input): #backward propagation
        vrainput = self.vraiinput(input)
        self.poids += self.cvcoef * (attendu - observe) * vrainput

    def validation(self, recherchee, valeur): #dire si le resultat est le bon
        return 1 if recherchee == valeur else 0

    def prediction(self, input):
        return self.fctactivescalier(self.produit(self.poids, self.vraiinput(input)))

    def entrainement(self, recherch): #on l'entraine self.iter fois sur la base avec recherch ce qu'on cherche a distinguer
        for n in range(self.iter):
            for fig in range(len(self.pix)):
                pred = self.prediction(self.pix[fig])
                self.changerpoids(self.validation(recherch, self.vales[fig]), pred, self.pix[fig])

    def tauxerreur(self, recherch, basepix, baseval):
        if not self.normal:
            correct = 0
            for i in range(len(basepix)):
                predator = self.prediction(basepix[i])
                correct += 1 if predator == self.validation(recherch, baseval[i]) else 0
                if self.dynamique:
                    self.changerpoids(self.validation(recherch, baseval[i]), predator, basepix[i])
            return round(100 - correct*100/len(basepix), 2)
        else:
            basepix = self.normaliserbase(basepix)
            correct = 0
            for i in range(len(basepix)):
                if self.boolbruitgaus:
                    predator = self.prediction(self.bruitgaussien(basepix[i]))
                elif self.boolbruitpix:
                    predator = self.prediction(self.bruitpixels(basepix[i], self.normal, self.saturation))
                else:
                    predator = self.prediction(basepix[i])
                correct += 1 if predator == self.validation(recherch, baseval[i]) else 0
                if self.dynamique:
                    self.changerpoids(self.validation(recherch, baseval[i]), predator, basepix[i])
            return round(100 - correct * 100 / len(basepix), 2)

    def testuneimage(self, image, recherch, vraivaleur):
        self.printcouleur(self.poids, "Poids")
        self.printcouleur(self.vraiinput(image), "Image")
        predator = self.prediction(image)
        if predator == 1:
            print(f"À mon avis ce chiffre est un {recherch}.")
        else:
            print(f"À mon avis ce chiffre ne ressemble point à un {recherch}.")
        print(f"En réalité la vraie valeur de cette image est {vraivaleur}.")
        if self.dynamique:
            self.changerpoids(self.validation(recherch, vraivaleur), predator, image)

def testuser():
    print(color.separer)
    cont = True
    while cont:
        print(color.separer)
        try:
            param = {"coefcv": float(input("Donnez moi un entier pour le taux d'apprentissage: ")), "iterations": int(input("Donnez moi un entier pour le nombre d'itérations: ")), "seuil": float(input("Donnez moi un entier pour le seuil: ")),
             "normal": bool(input("Donnez moi un boolean pour savoir si je normalise: ")), "apprentissagedynamique": bool(input("Donnez moi un boolean pour savoir si j'utilise l'apprentissage dynamique: ")),
             "bruitgaussien": bool(input("Donnez moi un boolean pour faire du bruit gaussien sur les images: ")), "pourcecarttype": int(input("Donnez moi un float pour l'écart-type de ce bruit: ")),
             "bruitsurpix": bool(input("Donnez moi un boolean pour faire du bruit sur des pixels des images: ")), "nbpixelsbruit": int(input("Donnez moi un entier pour le nombre de pixels à modifier: ")), "positionschoisies": False, "saturation": bool(input("Donnez moi un boolean pour savoir si je mets ces pixels en noir (True) ou en blanc (False): "))}

        except ValueError:
            print("Vous avez insérer des paramètres illégaux.")
            return

        try:
            perceptron = Perceptron(pixels, valeurs, **param)
        except:
            print("Vous avez insérer des paramètres illégaux.")
            return

        entrainesur = int(input("Sur quel chiffre voulez vous qu'il s'entraîne: "))
        perceptron.entrainement(entrainesur)

        print(color.UNDERLINE + f"Taux d'erreur:" + color.END, f"{perceptron.tauxerreur(entrainesur,qcmpix, qcmval)}", sep = " ")

        image = int(input("Donnez moi un entier entre 0 et 9999: "))

        if not 0<=image<=9999:
            print("Vous avez insérer des paramètres illégaux.")
            return

        perceptron.testuneimage(qcmpix[image], entrainesur, qcmval[image])
        print("Vous avez le droit de voir cette image dans les plots ainsi que mes poids.")

        cont = bool(input("Donnez False si vous voulez arrêter et True si vous voulez continuer: "))

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
   saut = "\n"
   separer = "_____________________________________________________________________________________________________________________________________________________________"

def testsbrutzero(qcmp, qcmv):
    Reference = Perceptron(pixels, valeurs, nbneurones = 784,
                iterations=1, coefcv = 0.1, seuil = 0, normal = False, apprentissagedynamique = False,
                bruitgaussien = False, pourcecarttype = 0,
                bruitsurpix = False, nbpixelsbruit = 0, positionschoisies = False, saturation = False)
    Reference.entrainement(0)

    print(color.separer)

    print("Tests pour rechercher le 0 dans une petite partie de la base de qcm. ", "Nous allons réaliser des tests avec différents parametres pour voir les changements.", sep = None)
    print("Nous allons changer les paramètres ceteris paribus pour voir l'effet et essayer de trouver les meilleurs paramètres.")
    print(color.separer)

    k, v = np.unique(qcmv, return_counts=True)
    nbchiffres = {int(k[i]): int(v[i]) for i in range(len(k))}
    print(color.BOLD + "Quelques informations sur la base de données utilisée: " + color.END + color.saut)
    print("Vous pouvez voir ici les quantités de chaque chiffre: ", f"{nbchiffres}", sep = None)
    print(f"La base de données sur laquelle il s'entraîne contient {len(valeurs)} photos")
    print(f"La base de données sur laquelle nous réalisons les tests contient {len(qcmv)} photos")
    print(color.separer)

    print(color.BLUE + color.BOLD + "Test de référence" + color.END + color.END, "\n")
    print("Commençons par tester avec tous les paramètres désactivés (si possible) pour avoir un point de référence: ",
          "Nous aurons donc tous les booléens en False, le taux d'apprentissage à 0.1, sans normalisation, seuil à 0 et une seule itération: ",
          color.UNDERLINE + "Taux d'erreur:" + color.END + f" {Reference.tauxerreur(0, qcmp, qcmv)}", sep = color.saut)
    Reference.printcouleur(Reference.poids, "Reference")
    print("Vous pouvez voir dans le plot de titre " + color.BOLD + "Reference" + color.END + " les poids que le réseau a obtenu.")

    print(color.separer)

    print(color.BLUE + color.BOLD + "Test sur différents chiffres" + color.END + color.END, "\n")

    for i in range(1,10):
        CH = Perceptron(pixels, valeurs, nbneurones=784,
                           iterations=1, coefcv=0.1, seuil=0, normal=False, apprentissagedynamique=False,
                           bruitgaussien=False, pourcecarttype=0,
                           bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
        CH.entrainement(i)
        print(color.UNDERLINE + f"Taux d'erreur sur {i}:" + color.END, f"{CH.tauxerreur(i,qcmp, qcmv)}", sep = " ")
        CH.printcouleur(CH.poids, f"Chiffre {i}")

    print("Vous pouvez voir les poids de chaque chiffre dans les plots.")

    print(color.separer)

    print(color.BLUE + color.BOLD + "Test avec et sans normalisation" + color.END + color.END, "\n")

    print("Test sans normalisation: " + f"{Reference.tauxerreur(0, qcmp, qcmv)}")

    Norm = Perceptron(pixels, valeurs, nbneurones=784,
                      iterations=1, coefcv=0.1, seuil=0, normal=True, apprentissagedynamique=False,
                      bruitgaussien=False, pourcecarttype=0,
                      bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
    Norm.entrainement(0)

    print(color.UNDERLINE + "Test avec normalisation:" + color.END, Norm.tauxerreur(0, qcmp, qcmv), sep = " ")

    print("Désormais nous utiliserons normé puisque les résultats sont meilleurs.")

    print(color.separer)

    print(color.BLUE + color.BOLD + "Test sur nombre itérations" + color.END + color.END, "\n")

    print("Le test de référence montre le taux d'erreur avec 1 itération donc nous le faisons avec des itérations de 5, 10, 15, 20, 50 et 100.")

    Testiter5 = Perceptron(pixels, valeurs, nbneurones=784,
                           iterations=5, coefcv=0.1, seuil=0, normal=True, apprentissagedynamique=False,
                           bruitgaussien=False, pourcecarttype=0,
                           bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
    Testiter5.entrainement(0)

    Testiter10 = Perceptron(pixels, valeurs, nbneurones=784,
                            iterations=10, coefcv=0.1, seuil=0, normal=True, apprentissagedynamique=False,
                            bruitgaussien=False, pourcecarttype=0,
                            bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
    Testiter10.entrainement(0)

    Testiter15 = Perceptron(pixels, valeurs, nbneurones=784,
                            iterations=15, coefcv=0.1, seuil=0, normal=True, apprentissagedynamique=False,
                            bruitgaussien=False, pourcecarttype=0,
                            bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
    Testiter15.entrainement(0)

    Testiter20 = Perceptron(pixels, valeurs, nbneurones=784,
                            iterations=20, coefcv=0.1, seuil=0, normal=True, apprentissagedynamique=False,
                            bruitgaussien=False, pourcecarttype=0,
                            bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
    Testiter20.entrainement(0)

    Testiter50 = Perceptron(pixels, valeurs, nbneurones=784,
                             iterations=50, coefcv=0.1, seuil=0, normal=True, apprentissagedynamique=False,
                             bruitgaussien=False, pourcecarttype=0,
                             bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
    Testiter50.entrainement(0)

    Testiter100 = Perceptron(pixels, valeurs, nbneurones=784,
                            iterations=100, coefcv=0.1, seuil=0, normal=True, apprentissagedynamique=False,
                            bruitgaussien=False, pourcecarttype=0,
                            bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
    Testiter100.entrainement(0)

    print(color.UNDERLINE + "Test avec 5 itérations:" + color.END, Testiter5.tauxerreur(0, qcmp, qcmv), sep = " ")
    Testiter5.printcouleur(Testiter5.poids, "5iter")

    print(color.UNDERLINE + "Test avec 10 itérations:" + color.END, Testiter10.tauxerreur(0, qcmp, qcmv), sep = " ")
    Testiter10.printcouleur(Testiter10.poids, "10iter")

    print(color.UNDERLINE + "Test avec 15 itérations:" + color.END, Testiter15.tauxerreur(0, qcmp, qcmv), sep = " ")
    Testiter15.printcouleur(Testiter15.poids, "15iter")

    print(color.UNDERLINE + "Test avec 20 itérations:" + color.END, Testiter20.tauxerreur(0, qcmp, qcmv), sep = " ")
    Testiter20.printcouleur(Testiter20.poids, "20iter")

    print(color.UNDERLINE + "Test avec 50 itérations:" + color.END, Testiter50.tauxerreur(0, qcmp, qcmv), sep=" ")
    Testiter50.printcouleur(Testiter50.poids, "50iter")

    print(color.UNDERLINE + "Test avec 100 itérations:" + color.END, Testiter100.tauxerreur(0, qcmp, qcmv), sep=" ")
    Testiter100.printcouleur(Testiter100.poids, "100iter")

    print("Vous pouvez voir comment changent les poids dans les plots avec un titre de la forme: (nombre)iter.")

    print("Désormais fixons les itérations à 15 qui a l'air d'être un bon compromis entre performance et temps d'exécution \n "
          "mais gardons en tête qu'à priori il pourrait y avoir de meilleures valeurs mais elles prennent plus de temps")

    print(color.separer)

    print(color.BLUE + color.BOLD + "Tests avec et sans apprentissage dynamique" + color.END + color.END, "\n")

    print("Ici on entend par apprentissage dynamique le fait qu'il s'entraine même sur les images de test.")

    print("Test sans apprentissage dynamique: " + f"{Testiter15.tauxerreur(0, qcmp, qcmv)}")

    Dynam = Perceptron(pixels, valeurs, nbneurones=784,
                            iterations=15, coefcv=0.1, seuil=0, normal=True, apprentissagedynamique=True,
                            bruitgaussien=False, pourcecarttype=0,
                            bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
    Dynam.entrainement(0)

    print(color.UNDERLINE + "Test avec apprentissage dynamique:" + color.END, Dynam.tauxerreur(0, qcmp, qcmv), sep=" ")

    print("Désormais nous ne gardons pas l'apprentissage dynamique qui semble empirer le taux d'erreur, même si sûrement il peut être bien dans certaines circonstances.")

    print(color.separer)

    print(color.BLUE + color.BOLD + "Tests avec différents seuils" + color.END + color.END, "\n")

    seuils = [0.001, 0.01, 0.1, 0.15, 0.2]
    for i in seuils:
        SE = Perceptron(pixels, valeurs, nbneurones=784,
                                iterations=15, coefcv=0.1, seuil=i, normal=True, apprentissagedynamique=False,
                                bruitgaussien=False, pourcecarttype=0,
                                bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
        SE.entrainement(0)
        print(color.UNDERLINE + f"Taux d'erreur avec un seuil de {i}:" + color.END, f"{SE.tauxerreur(0, qcmp, qcmv)}", sep=" ")

    print("Désormais gardons un seuil de 0.15 qui a l'air d'être bien, même si surement il en existe de meilleur.")

    print(color.separer)

    print(color.BLUE + color.BOLD + "Tests avec différents taux d'apprentissage" + color.END + color.END, "\n")

    tauxapp = [0.001, 0.01, 0.1, 0.15, 0.2, 0.5]
    for i in tauxapp:
        TA = Perceptron(pixels, valeurs, nbneurones=784,
                        iterations=15, coefcv=i, seuil=0.15, normal=True, apprentissagedynamique=False,
                        bruitgaussien=False, pourcecarttype=0,
                        bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
        TA.entrainement(0)
        print(color.UNDERLINE + f"Taux d'erreur avec un taux d'apprentissage de {i}:" + color.END, f"{TA.tauxerreur(0, qcmp, qcmv)}", sep=" ")

    print("Désormais gardons un taux d'apprentissage de 0.1 qui a l'air d'être bien.")

    print(color.separer)

    print(color.BLUE + color.BOLD + "Tests avec un bruit gaussien avec différents écart-types" + color.END + color.END, "\n")

    ecarts = [0, 5, 10, 15, 30]
    for i in ecarts:
        BG = Perceptron(pixels, valeurs, nbneurones=784,
                        iterations=15, coefcv=0.1, seuil=0.15, normal=True, apprentissagedynamique=False,
                        bruitgaussien=True, pourcecarttype=i,
                        bruitsurpix=False, nbpixelsbruit=0, positionschoisies=False, saturation=False)
        BG.entrainement(0)
        print(color.UNDERLINE + f"Taux d'erreur avec un écart-type de {i}:" + color.END,
              f"{BG.tauxerreur(0, qcmp, qcmv)}", sep=" ")

    print(color.separer)

    print(color.BLUE + color.BOLD + "Tests avec un bruit sur les pixels variant le nombre de pixels modifiés" + color.END + color.END,
          "\n")

    nb = [0, 50,100,200,300]
    for i in nb:
        BP = Perceptron(pixels, valeurs, nbneurones=784,
                        iterations=15, coefcv=0.1, seuil=0.15, normal=True, apprentissagedynamique=False,
                        bruitgaussien=False, pourcecarttype=0,
                        bruitsurpix=True, nbpixelsbruit=i, positionschoisies=False, saturation=True)
        BP.entrainement(0)
        print(color.UNDERLINE + f"Taux d'erreur avec {i} pixels modifiés:" + color.END,
              f"{BP.tauxerreur(0, qcmp, qcmv)}", sep=" ")

    print(color.separer)

    print("Ici, nous avons donc trouvé une disposition possible des paramètres mais il y en a plein \n"
          "mais dans tous les cas les bruits vont affecter la qualité de la prédiction.")

    param = {"iterations":20, "coefcv":0.001, "seuil":0.01, "normal":True, "apprentissagedynamique":True, "bruitgaussien":True, "pourcecarttype":15, "bruitsurpix":True, "nbpixelsbruit":100, "positionschoisies":False, "saturation":True}
    print("Par exemple une bonne configuration contre le bruit peut être:",
          param)

    par = Perceptron(pixels, valeurs, **param)
    par.entrainement(0)
    print(color.UNDERLINE + f"Taux d'erreur:" + color.END, f"{par.tauxerreur(0, qcmp, qcmv)}", sep=" ")


    print("Vous avez la possibilité de tester vous-même en mettant les paramètre de votre choix dans l'input suivant.")

    testuser()

# testsbrutzero(qcmpix, qcmval)

# testuser()

