from utilities import init
from error_functions import *
from KNN import knn
from SVM import SVM
from DT import DT
init()

print("K Nearest Neighbours Test--------------------------------------")
#KFoldValidation(10, knn)
bootstrap(10, knn)

print("Support Vector Machine Test--------------------------------------")
#KFoldValidation(10, SVM)
bootstrap(10, SVM)

print("Decision Tree Test--------------------------------------")
#KFoldValidation(10, DT)
bootstrap(10, DT)

'''
'''