import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from numpy.lib.type_check import real
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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

def method(X_train,X_test,X_train_data,X_test_data):
    print(X_train[0]) 
    print(X_train_data[0]) 
    print("###########")
    print(X_test[0]) 
    print(X_test_data[0]) 
    print("########################")

def get_partitions(K,method): # K-folds N<K
    data = read_data() #####################################
    kf = KFold(n_splits=K)
    X_train = []
    X_test = []
    X_train_data =[]
    X_test_data = []
    for train_index, test_index in kf.split(data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train_data =[]
        X_test_data = []
        X_train, X_test = data[train_index], data[test_index]
        for item in X_train:
            X_train_data.append(item["data"])
        for item in X_test:
            X_test_data.append(item["data"])
        ##here goes the algorithm
        method(X_train,X_test,X_train_data,X_test_data)


#get_partitions(2,method)

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