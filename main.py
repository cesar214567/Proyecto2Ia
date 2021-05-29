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

#save_data()

def read_data():
    data = np.load("db.npy",allow_pickle=True)
    return data

'''def method(X_train,X_test,X_train_data,X_test_data):
    print(X_train[0]) 
    print(X_train_data[0]) 
    print("###########")
    print(X_test[0]) 
    print(X_test_data[0]) 
    print("########################")
'''

def get_partitions(K,method): # K-folds N<K
    data = read_data() #####################################
    data_train, data_test = train_test_split(data, test_size=0.3)
    kf = KFold(n_splits=K, random_state=100, shuffle=True)
    error = 0
    for train_index, test_index in kf.split(data_train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        train_data =[]
        test_data = []
        train, test = data_train[train_index], data_train[test_index]
        for item in train:
            train_data.append(item["data"])
        for item in test:
            test_data.append(item["data"])
        ##here goes the algorithm
        temp = method(train,test,train_data,test_data)
        print(temp)
        error+=temp
    print("Error final: ", error/K)
    train_data =[]
    test_data = []
    for item in data_train:
        train_data.append(item["data"])
    for item in data_test:
        test_data.append(item["data"])
    print("Error model: ", method(data_train,data_test,train_data,test_data))
    


#get_partitions(2,knn)

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
