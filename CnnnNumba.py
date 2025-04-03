#GENERAL
import time

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

val, pix, qcmval, qcmpix, pixelsconv, qcmpixconv = takeinputs()

convlay = [(1, "input"), (10, "relu", True)]

lay = [(64, "sigmoid"), (10, "softmax")]

