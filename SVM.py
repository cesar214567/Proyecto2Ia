from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from utilities import getY
from utilities import confuse_matrix

@ignore_warnings(category=ConvergenceWarning)
def SVM(X_train,X_test,X_train_data,X_test_data,option,method,i):
    y_train = getY(X_train)
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=option))
    clf.fit(X_train_data, y_train)    
    y_test = getY(X_test)

    #if show == True:
    result = clf.predict(X_test_data)
    #print(option)
    dic =  { 10:1, 1:2, 0.1:3, 0.01:4, 0.001:5, 0.0001:6, 0.00001:7}


    confuse_matrix(y_test,result, str(method) + '/SVM/' + str(i)  + '_' +  str(dic[option]))

    return clf.score(X_test_data,y_test)
