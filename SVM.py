from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from KNN import save_data
from utilities import getY
from utilities import confuse_matrix

def SVM(X_train,X_test,X_train_data,X_test_data,method=None,show=False): 
    y_train = getY(X_train)
    clf = make_pipeline(StandardScaler(),
                        LinearSVC(random_state=0, tol=1e-5))
    clf.fit(X_train_data, y_train)    
    y_test = getY(X_test)

    if show == True:
        result = clf.predict(X_test_data)
        confuse_matrix(y_test,result, "SVM_"+method)

    return clf.score(X_test_data,y_test)