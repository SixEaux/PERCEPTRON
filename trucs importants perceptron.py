import pickle

valeur = []
pixels = []

import csv
with open('mnist_train.csv', mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for idx, row in enumerate(reader):
        if row and idx!=0:
            valeur.append(int(row[0]))
            p = [int(i) for i in row[1:]]
            pixels.append(p)


with open('valeursentraine', 'wb') as f:
    pickle.dump(valeur, f)

with open('pixelsentraine', 'wb') as f:
    pickle.dump(pixels, f)
