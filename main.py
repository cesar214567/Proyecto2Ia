import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from numpy.lib.type_check import real
from sklearn.model_selection import train_test_split

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
def getType(data):
    words = data.split("/")
    return words[1]


def save_data():
    ds_train = []
    fl = open("db.txt", "r")
    cont =0
    
    for line in fl:
        real_url = line[:-1]
        img = imread(real_url)
        coeffs2 = pywt.dwt2(img,'haar')
        LL, (LH, HL, HH) = coeffs2
        coeffs2 = pywt.dwt2(LL,'haar')
        LL, (LH, HL, HH) = coeffs2
        coeffs2 = pywt.dwt2(LL,'haar')
        LL, (LH, HL, HH) = coeffs2
        LL = LL.flatten()        
        dict = {"id":cont,"type":getType(real_url),"url":real_url , "data":LL}
        ds_train.append(dict)
        cont+=1
    np.save("db.npy",ds_train,allow_pickle=True)
    return ds_train

#test();

def read_data():
    data = np.load("db.npy",allow_pickle=True)
    return data

def get_partitions(N,K): # K-folds N<K
    data = read_data()
    size = len(data)
    train = []
    test = []
    train_data =[]
    test_data = []
    for i in range(size):
        if (i>=size*N/K and i<size*(N+1)/K):
            test.append(data[i])
            test_data.append(data[i]["data"])
        else:
            train.append(data[i])
            train_data.append(data[i]["data"])
    
    return train,test,train_data,test_data

#save_data()

#data=read_data()

train,test,train_data,test_data = get_partitions(1,7)

for i in test:
    print(i)



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