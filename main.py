import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

import pywt
import pywt.data

#A = imread("CK+48/sadness/S115_004_00000016.png")
#original = A
## Load image
##original = pywt.data.camera()
#
## Wavelet transform of image, and plot approximation and details
#titles = ['Approximation', ' Horizontal detail',
#          'Vertical detail', 'Diagonal detail']
#coeffs2 = pywt.dwt2(original, 'haar')
#LL, (LH, HL, HH) = coeffs2
#print(LL.flatten())
#print("--------")
#coeffs2 = pywt.dwt2(LL, 'haar')
#LL, (LH, HL, HH) = coeffs2
#print(LL.flatten())
#
def read_data():
    ds_train = []
    fl = open("db.txt", "r")
    for line in fl:
        img = imread(line)
        coeffs2 = pywt.dwt2(img,'haar')
        LL, (LH, HL, HH) = coeffs2
        coeffs2 = pywt.dwt2(LL,'haar')
        LL, (LH, HL, HH) = coeffs2
        ds_train.append(LL)
    return ds_train

        
read_data()
#fig = plt.figure(figsize=(12, 3))
#for i, a in enumerate([LL, LH, HL, HH]):
#    ax = fig.add_subplot(1, 4, i + 1)
#    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#    ax.set_title(titles[i], fontsize=10)
#    ax.set_xticks([])
#    ax.set_yticks([])
#
#fig.tight_layout()
#plt.show()
#plt.clf()
#
#
#
#coeffs2 = pywt.dwt2(LL, 'bior1.3')
#LL, (LH, HL, HH) = coeffs2
#print(LL)
#
#fig = plt.figure(figsize=(12, 3))
#for i, a in enumerate([LL, LH, HL, HH]):
#    ax = fig.add_subplot(1, 4, i + 1)
#    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
#    ax.set_title(titles[i], fontsize=10)
#    ax.set_xticks([])
#    ax.set_yticks([])
#
#fig.tight_layout()
#plt.show()