from utilities import read_data
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
        return [10, 12, 14, 16, 20, 24, 28, 32, 36]
    elif(method_name == "SVM"):
        return [1e-2, 1e-3, 1e-4, 1e-5]


def fold(data,kf, data_train, K, option, method):
    accuracy = 0
    for train_index, test_index in kf.split(data_train):
        train, test = data[train_index], data[test_index]
        train_data,test_data = distribute_data(train,test)
        temp = method(train,test,train_data,test_data, option)
        accuracy+=temp
    return accuracy/K

def KFoldValidation(K,method): # K-folds N<K
    data = read_data()
    method_name = method.__name__
    data_train, data_test = train_test_split(data, test_size=0.3)
    kf = KFold(n_splits=K)
    options = set_options(method_name)
    accuracies = []
    errors = []
    for option in options:
        temp_accuracy = fold(data, kf, data_train, K, option, method)
        print(method_name, "KFold", " - option ", option)
        print("Final estimator accuracy", temp_accuracy)
        print("Final estimator error", 1-temp_accuracy)
        accuracies.append(temp_accuracy)
        errors.append(1-temp_accuracy)
        
    '''train_data,test_data = distribute_data(data_train,data_test)
    accuracy = method(data_train,data_test,train_data,test_data,"KFold",True)
    print("Model accuracy: ", accuracy)
    print("Model error: ", 1-accuracy)'''
    return [accuracies, errors]
    
def sampling(data, data_train, K, option, method):
    accuracy = 0
    for rep in range(K):
        boot_sample = resample(data_train, n_samples=len(data_train))
        test_set = [x for x in data if x not in boot_sample]
        train_data,test_data = distribute_data(boot_sample,test_set)
        temp = method(boot_sample,test_set,train_data,test_data, option)
        accuracy+=temp
    return accuracy / K
    

def bootstrap(K,method): # K-folds N<K
    data = read_data()
    method_name = method.__name__
    data_train, data_test = train_test_split(data, test_size=0.3)
    options = set_options(method_name)
    accuracies = []
    errors = []
    for option in options:
        temp_accuracy = sampling(data, data_train, K, option, method)
        print(method_name, "bootstrap" + " - option ", option)
        print("Final estimator accuracy", temp_accuracy)
        print("Final estimator error", 1-temp_accuracy)
        accuracies.append(temp_accuracy)
        errors.append(1-temp_accuracy)

    '''train_data,test_data = distribute_data(data_train,data_test)
    accuracy = method(data_train,data_test,train_data,test_data,"Bootstrap",True)
    print("Model accuracy: ", accuracy)
    print("Model error: ", 1-accuracy)    
    '''
    return [accuracies, errors]
