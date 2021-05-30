import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
from sklearn import metrics
import seaborn as sn
import pywt
import pywt.data
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

def getY(X):
    y = []
    for i in X:
        y.append(i['type'])
    return y

def getType(data):
    words = data.split("/")
    return words[1]


def save_data():
    data = []
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
        data.append(dict)
        cont+=1
    np.save("db.npy",data,allow_pickle=True)
    return data

def read_data():
    data = np.load("db.npy",allow_pickle=True)
    arr = []
    for d in data:
        arr.append(d['data'])
    np.savetxt("dbknn.csv",arr, delimiter=",")
    return data

def init(read_file=False):
    if(read_file):
        save_data()
    rs = RandomState(MT19937(SeedSequence(123456789)))
    np.random.seed(3)

def confuse_matrix(y_ts, results, name):
    labels = ["fear", "anger", "contempt", "happy", "disgust","sadness","surprise"]
    confusion_matrix = metrics.confusion_matrix(y_ts, results, labels=labels)
    df_cm = pd.DataFrame(confusion_matrix, index = labels,
                    columns = labels)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig("confusion_matrixes/"+name)
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

read_data()