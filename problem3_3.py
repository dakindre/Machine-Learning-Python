import sys
import argparse
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out





def SVM(value):
    
    #Create Training and Test 60/40 stratified
    X_train, X_test, Y_train, Y_test = train_test_split(value.x, value.y, test_size=0.4, stratify=value.y)

    #Set param for SVM Linear and run fit method to find model using training data
    svmlp = {'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
    SVML = GridSearchCV(SVC(), svmlp, cv=5, refit=True)
    SVML.fit(X_train, Y_train)
    testScoreSMLP = SVML.score(X_test, Y_test)
    print('svm_linear ', 'best_training_score: ', SVML.best_score_, 'actual_test_score: ', testScoreSMLP)
    
    #Set param for SVM Polynomial and run fit method to find model using training data
    svmpp = {'kernel': ['poly'], 'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 0.5]}
    SVMP = GridSearchCV(SVC(), svmpp, cv=5, refit=True)
    SVMP.fit(X_train, Y_train)
    testScoreSMP = SVMP.score(X_test, Y_test)
    print('svm_poly ', 'best_training_score: ', SVMP.best_score_, 'actual_test_score: ', testScoreSMP)

    #Set param for SVM RBF and run fit method to find model using training data
    svmrp = {'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}
    SVMRBF = GridSearchCV(SVC(), svmrp, cv=5, refit=True)
    SVMRBF.fit(X_train, Y_train)
    testScoreSMRBF = SVMRBF.score(X_test, Y_test)
    print('svm_rbf ', 'best_training_score: ', SVMRBF.best_score_, 'actual_test_score: ', testScoreSMRBF)

    #Set param for Logistic Regression and run fit method to find model using training data
    inputparam = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
    LR = GridSearchCV(LogisticRegression(), inputparam, cv=5, refit=True)
    LR.fit(X_train, Y_train)
    testScoreLR = LR.score(X_test, Y_test)
    print('logistic_regression ', 'best_training_score: ', LR.best_score_, 'actual_test_score: ', testScoreLR)

    #Set param for K nearest neighbors and run fit method to find model using training data
    inputparam = {'n_neighbors': list(range(1, 50)), 'leaf_size': [5,10,15,20,25,30,35,40,45,50,55,60]}
    KNN = GridSearchCV(KNeighborsClassifier(), inputparam, cv=5, refit=True)
    KNN.fit(X_train, Y_train)
    testScoreKNN = KNN.score(X_test, Y_test)
    print('k-nearest_neighbors ', 'best_training_score: ', KNN.best_score_, 'actual_test_score: ', testScoreKNN)

    #Set param for Decision Tree Classification and run fit method to find model using training data
    inputparam = {'max_depth': list(range(1, 50)), 'min_samples_split': [2,3,4,5,6,7,8,9,10]}
    DT = GridSearchCV(DecisionTreeClassifier(), inputparam, cv=5, refit=True)
    DT.fit(X_train, Y_train)
    testScoreDT = DT.score(X_test, Y_test)
    print('decision_tree_class ', 'best_training_score: ', DT.best_score_, 'actual_test_score: ', testScoreDT)

    #Set param for Random Forest Classification and run fit method to find model using training data
    inputparam = {'max_depth': list(range(1, 50)), 'min_samples_split': [2,3,4,5,6,7,8,9,10]}
    DT = GridSearchCV(RandomForestClassifier(), inputparam, cv=5, refit=True)
    DT.fit(X_train, Y_train)
    testScoreDT = DT.score(X_test, Y_test)
    print('random_forest_class ', 'best_training_score: ', DT.best_score_, 'actual_test_score: ', testScoreDT)

    
class inOut:
    def __init__(self, inStr, outStr):
        self.inStr = inStr
        self.outStr = outStr
        self.x = np.empty((0,2))
        self.y = np.empty((0,1))
        self.generateInput()

    def generateInput(self):
        #Initialize Temp Empty Array/List
        self.tempX = np.empty((0,2))
        tempY = []

        #Read data in from csv file into X and Y
        with open(self.inStr) as csvfile:
            inputCSV = csv.reader(csvfile, delimiter=',')
            next(inputCSV)
            for row in inputCSV:
                self.x = np.append(self.x, np.array([[float(row[0]), float(row[1])]]), axis=0)
                tempY.append(float(row[2]))
            self.y = np.asarray(tempY)
                
    def generateOutput(self, output):
        with open(self.outStr, 'a') as csvFile:
            fileWriter = csv.writer(csvFile, lineterminator='\n')
            fileWriter.writerow(output)

    
        
        
                
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=str)
    parser.add_argument('o', type=str)

    args = parser.parse_args()

    value = inOut(args.i, args.o)

    SVM(value)



if __name__ == "__main__":
	main()
