import pickle
import numpy as np

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps


def takeinputs():

    with open('Datas/valeursentraine', 'rb') as f:
        valeurs = np.array(pickle.load(f))

    with open('Datas/pixelsentraine', 'rb') as f:
        p = pickle.load(f)
        pixels = np.array(p).T
        pixelsconv = [np.array(a).reshape((28,28)) for a in p]

    with open('Datas/testval', 'rb') as f:
        qcmval = pickle.load(f)

    with open('Datas/testpix', 'rb') as f:
        qp = pickle.load(f)
        qcmpix = np.array(qp).T
        qcmpixconv = [np.array(a).reshape((28,28)) for a in qp]

    perm = np.random.permutation(pixels.shape[1])

    pixmelange = pixels[:, perm]
    valmelange = valeurs[perm]

    pixmelangeconv = list(map(pixelsconv.__getitem__, perm))

    return valmelange, pixmelange, qcmval, qcmpix, pixmelangeconv, qcmpixconv

class Draw:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Paint")

        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.pack()

        self.dessine = False

        self.posx, self.posy = None,None

        self.butons = []

        self.image = Image.new("L", (280, 280), 255)
        self.drawing = ImageDraw.Draw(self.image)

        self.pixels = None

        self.creerboutons()

        self.root.mainloop()

    def creerboutons(self):
        imprimer = tk.Button(self.root, text="Print", command=self.imprime)
        imprimer.pack(side=tk.LEFT)

        fermer = tk.Button(self.root, text="Fermer", command=self.root.destroy)
        fermer.pack(side=tk.LEFT)

        self.butons.append(imprimer)
        self.butons.append(fermer)


        self.canvas.bind("<Button-1>", self.commence)
        self.canvas.bind("<ButtonRelease-1>", self.arret)
        self.canvas.bind("<B1-Motion>", self.draw)

    def commence(self, event):
        self.dessine = True
        self.posx, self.posy = event.x, event.y

    def arret(self):
        self.dessine = False

    def draw(self, event):
        if self.dessine:
            x, y = event.x, event.y

            self.canvas.create_line((self.posx, self.posy, x, y), fill="white", width=10)

            self.drawing.line([self.posx, self.posy, x, y], fill=0, width=8)

            self.posx, self.posy = x, y



    def imprime(self):
        im = self.image.resize((28, 28), Image.Resampling.LANCZOS) #.convert("L")

        im = ImageOps.invert(im)

        # im.show()

        self.pixels = np.array(im.getdata()).reshape(-1,1)


        self.root.destroy()