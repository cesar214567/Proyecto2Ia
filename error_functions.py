from utilities import read_data
from utilities import read_data_less
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import csv
import numpy as np

import matplotlib.pyplot as plt
from numpy import mean, var
def distribute_data(train,test):
    train_data =[]
    test_data = []
    for item in train:
        train_data.append(item["data"])
    for item in test:
        test_data.append(item["data"])
    return train_data,test_data

def set_options(method_name):
    if(method_name == "knn"):
        return [1, 2, 4, 6, 8, 10, 12, 14]
    elif(method_name == "DT"):
        return [2, 4, 8, 10, 14, 16, 24, 32, 36]
    elif(method_name == "SVM"):
        return [1e1, 1e-1, 1, 1e-2, 1e-3, 1e-4, 1e-5]


def fold(data,kf, data_train, K, option, method):
    accuracies = []
    for train_index, test_index in kf.split(data_train):
        train, test = data[train_index], data[test_index]
        train_data,test_data = distribute_data(train,test)
        temp = method(train,test,train_data,test_data, option, "KFold")
        accuracies.append(temp)
    promedio = mean(accuracies)
    varianza = var(accuracies)
    return [promedio, varianza]

def plot(options,accuracies,method_name,type):
    plt.axis([0,options[len(options)-1]+1,0,1])
    plt.plot(options,accuracies,'*')
    #plt.show()
    plt.savefig('results/'+method_name+'-'+type+'.png')
    plt.clf()

def KFoldValidation(K,method): # K-folds N<K
    data = read_data_less()
    method_name = method.__name__
    data_train, data_test = train_test_split(data, test_size=0.3)
    kf = KFold(n_splits=K)
    options = set_options(method_name)
    accuracies = []
    variances = []
    errors = []
    for option in options:
        results = fold(data, kf, data_train, K, option, method)
        print(method_name, "KFold", " - option ", option)
        print("Final estimator accuracy", results[0])
        print("Final estimator error", 1-results[0])
        accuracies.append(results[0])
        variances.append(results[1])
        errors.append(1-results[0])
    if(method_name == "SVM"):
        options = [1, 0, -1,- 2, -3, -4, -5]
    plot(options,accuracies,method_name,'Kfold')
    '''train_data,test_data = distribute_data(data_train,data_test)
    accuracy = method(data_train,data_test,train_data,test_data,"KFold",True)
    print("Model accuracy: ", accuracy)
    print("Model error: ", 1-accuracy) '''
    return [accuracies, errors]
    
def sampling(data, data_train, K, option, method):
    accuracies = []
    for rep in range(K):
        boot_sample = resample(data_train, n_samples=len(data_train))
        test_set = [x for x in data if x not in boot_sample]
        train_data,test_data = distribute_data(boot_sample,test_set)
        temp = method(boot_sample,test_set,train_data,test_data, option, "bootstrap")
        accuracies.append(temp)
    promedio = mean(accuracies)
    varianza = var(accuracies)
    return [promedio, varianza]

def bootstrap(K,method): # K-folds N<K
    data = read_data_less()
    method_name = method.__name__
    data_train, data_test = train_test_split(data, test_size=0.3)
    options = set_options(method_name)
    accuracies = []
    variances = []
    errors = []
    for option in options:
        results = sampling(data, data_train, K, option, method)
        print(method_name, "bootstrap" + " - option ", option)
        print("Final estimator accuracy", results[0])
        print("Final estimator variance", results[1])
        print("Final estimator error", 1-results[0])
        accuracies.append(results[0])
        variances.append(results[1])
        errors.append(1-results[0])
    if(method_name == "SVM"):
        options = [1, 0, -1,- 2, -3, -4, -5]
    plot(options,accuracies,method_name,'bootstrap')

    '''train_data,test_data = distribute_data(data_train,data_test)
    accuracy = method(data_train,data_test,train_data,test_data,"Bootstrap",True)
    print("Model accuracy: ", accuracy)
    print("Model error: ", 1-accuracy)    
    '''
    
    return [accuracies, errors]
