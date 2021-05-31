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
        return [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]


def fold(data,kf, data_train, K, option, method):
    accuracies = []
    best_i = 0
    best = 0
    i = 0
    for train_index, test_index in kf.split(data_train):
        train, test = data[train_index], data[test_index]
        train_data,test_data = distribute_data(train,test)
        temp = method(train,test,train_data,test_data, option, "KFold" , i)
        if temp > best:
            best_i = i
            best = temp
        accuracies.append(temp)
        i += 1
    promedio = mean(accuracies)
    varianza = var(accuracies)
    return [promedio, varianza, best,best_i]

def plot(options,accuracies,method_name,type):
    plt.axis([0,options[len(options)-1]+1,0,1])
    plt.plot(options,accuracies,'*')
    #plt.show()
    plt.savefig('results/'+method_name+'-'+type+'.png')
    plt.clf()

def KFoldValidation(K,method,show=False): # K-folds N<K
    data = read_data_less()
    method_name = method.__name__
    #data_train, data_test = train_test_split(data, test_size=0.3)
    data_train = data
    kf = KFold(n_splits=K,shuffle=True)
    options = set_options(method_name)
    accuracies = []
    variances = []
    errors = []
    best = 0
    best_option= 0
    best_i = 0
    for option in options:
        results = fold(data, kf, data_train, K, option, method)
        if results[2]> best:
            best = results[2]
            best_option = option
            best_i = results[3]
        if show:
            print(method_name, "KFold", " - option ", option)
            print("Final estimator accuracy", results[0])
            print("Final estimator variance", results[1])
            print("Final estimator error", 1-results[0])
        accuracies.append(results[0])
        variances.append(results[1])
        errors.append(1-results[0])
    if(method_name == "SVM"):
        options = [1, 0, -1,- 2, -3, -4, -5]
    plot(options,accuracies,method_name,'Kfold')
    print("Kfold----- ")
    print("Best i: ",best_i, " best option: ",best_option)

    '''
    best = 0
    best_option= 0
    for option in options:
        train_data,test_data = distribute_data(data_train,data_test)
        accuracy = method(data_train,data_test,train_data,test_data, option,"KFold",True)
        if accuracy > best:
            best = accuracy
            best_option = option
        print("Model accuracy: ", accuracy)
        print("Model error: ", 1-accuracy) 
    print("Mode best option: ", best_option)
    '''
    return [accuracies, errors]
    
def sampling(data, data_train, K, option, method):
    accuracies = []
    best = 0
    best_K = 0
    for rep in range(K):
        boot_sample = resample(data_train, n_samples=len(data_train))
        test_set = [x for x in data if x not in boot_sample]
        train_data,test_data = distribute_data(boot_sample,test_set)
        temp = method(boot_sample,test_set,train_data,test_data, option, "Bootstrap" ,rep)
        if temp > best:
            best_K = rep
            best = temp
        accuracies.append(temp)
    promedio = mean(accuracies)
    varianza = var(accuracies)
    return [promedio, varianza,best ,best_K]

def bootstrap(K,method,show= False): # K-folds N<K
    data = read_data_less()
    method_name = method.__name__
   # data_train, data_test = train_test_split(data, test_size=0.3)
    data_train = data
    options = set_options(method_name)
    accuracies = []
    variances = []
    errors = []
    best = 0
    best_option= 0
    best_i = 0
    for option in options:
        results = sampling(data, data_train, K, option, method)
        if results[2]> best:
            best = results[2]
            best_option = option
            best_i = results[3]
        if show:
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
    print("Boostrap----- ")
    print("Best K: ",best_i, " best option: ",best_option)
    
    '''train_data,test_data = distribute_data(data_train,data_test)
    accuracy = method(data_train,data_test,train_data,test_data,"Bootstrap",True)
    print("Model accuracy: ", accuracy)
    print("Model error: ", 1-accuracy)    
    '''
    
    return [accuracies, errors]
