from utilities import read_data
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def KFoldValidation(K,method): # K-folds N<K
    data = read_data()
    data_train, data_test = train_test_split(data, test_size=0.3)
    kf = KFold(n_splits=K)
    accuracy = 0
    for train_index, test_index in kf.split(data_train):
        train_data =[]
        test_data = []
        train, test = data_train[train_index], data_train[test_index]
        for item in train:
            train_data.append(item["data"])
        for item in test:
            test_data.append(item["data"])
        temp = method(train,test,train_data,test_data)
        print(temp)
        accuracy+=temp
    accuracy /= K
    print("Final estimator accuracy: ", accuracy)
    print("Final estimator error: ", 1-accuracy)
    train_data =[]
    test_data = []
    for item in data_train:
        train_data.append(item["data"])
    for item in data_test:
        test_data.append(item["data"])
    accuracy = method(data_train,data_test,train_data,test_data,"KFold",True)
    print("Model accuracy: ", accuracy)
    print("Model error: ", 1-accuracy)
    return [accuracy, 1-accuracy]
    

def bootstrap(K,method): # K-folds N<K
    data = read_data()
    data_train, data_test = train_test_split(data, test_size=0.3)
    accuracy = 0
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
        accuracy+=temp
    accuracy /= K
    print("Final estimator accuracy: ", accuracy)
    print("Final estimator error: ", 1-accuracy)
    train_data =[]
    test_data = []
    for item in data_train:
        train_data.append(item["data"])
    for item in data_test:
        test_data.append(item["data"])
    accuracy = method(data_train,data_test,train_data,test_data,"Bootstrap",True)
    print("Model accuracy: ", accuracy)
    print("Model error: ", 1-accuracy)    
    return [accuracy, 1-accuracy]