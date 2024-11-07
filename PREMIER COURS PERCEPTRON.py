class Perceptron:
    def __init__(self,nbneurones, coefcv = 0.1):
        self.nb = nbneurones
        self.neurones = [(None, 255) for i in range(nbneurones)]
        self.cvcoef = coefcv

    def somme(self):
        #produit
        pass

    def fctactiv(self):
        #obtenir 1 ou 0 si chiffre qu'on cherche ou pas
        pass

    def changerpoids(self):
        #si bomne reponse on garde les poids, si erreur pensant que c'est le chiffre attendu - reponse = -1 sinon inverse = 1
        pass




