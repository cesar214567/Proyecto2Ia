import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from numpy.lib.type_check import real
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
import pandas as pd
import seaborn as sn
import pywt
import pywt.data
from sklearn.utils import resample
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence


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

def read_data():
    data = np.load("db.npy",allow_pickle=True)
    return data

def get_partitions(K,method): # K-folds N<K
    data = read_data() #####################################
    data_train, data_test = train_test_split(data, test_size=0.3)
    kf = KFold(n_splits=K)
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
    print("Error model: ", method(data_train,data_test,train_data,test_data,True))
    

def confuse_matrix(y_ts, results):
    labels = ["fear", "anger", "contempt", "happy", "disgust","sadness","surprise"]
    confusion_matrix = metrics.confusion_matrix(y_ts, results, labels=labels)
    df_cm = pd.DataFrame(confusion_matrix, index = labels,
                    columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(results)):
        if(results[i] == 0 and results[i]==y_ts[i]):
            TP += 1
        elif(results[i] == 0 and results[i]!=y_ts[i]):
            FP += 1
        elif(results[i] != 0 and results[i]==y_ts[i]):
            FN += 1
        elif(results[i] != 0 and results[i]!=y_ts[i]):
            TN += 1
    return [TP, FP, TN, FN]

def bootstrap(K,method): # K-folds N<K
    data = read_data() #####################################
    data_train, data_test = train_test_split(data, test_size=0.3)
    error = 0
    for rep in range(K):
        boot_sample = resample(data_train, n_samples=len(data_train))
        test_set = [x for x in data if x not in boot_sample]
        train_data =[]
        test_data = []
        for item in boot_sample:
            train_data.append(item["data"])
        for item in test_set:
            test_data.append(item["data"])
        temp = method(boot_sample,test_set,train_data,test_data)
        print(temp)
        error+=temp
    print("Error final: ", error/K)
    train_data =[]
    test_data = []
    for item in data_train:
        train_data.append(item["data"])
    for item in data_test:
        test_data.append(item["data"])
    print("Error model: ", method(data_train,data_test,train_data,test_data,True))




