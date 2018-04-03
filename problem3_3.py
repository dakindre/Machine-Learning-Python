import sys
import argparse
import csv
import numpy as np
from sklearn import svm
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


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
    

def SGD(value, a, ni):
    
    #Gradient Descent Method
    clf = linear_model.SGDRegressor(alpha=a, n_iter = ni)
    clf.fit(value.x, value.y, coef_init = [0,0,0])

    #Collect Values and output to csv file
    outputList = [a, ni, clf.coef_[0], clf.coef_[1], clf.coef_[2]]
    value.generateOutput(outputList)
    
    
    
class inOut:
    def __init__(self, inStr, outStr):
        self.inStr = inStr
        self.outStr = outStr
        self.x = np.empty((0,3))
        self.y = np.empty((0,1))
        self.generateInput()
        self.scale()

    def generateInput(self):
        #Initialize Temp Empty Array/List
        self.tempX = np.empty((0,2))
        tempY = []

        #Read data in from csv file into X and Y
        with open(self.inStr) as csvfile:
            inputCSV = csv.reader(csvfile, delimiter=',')
            for row in inputCSV:
                self.tempX = np.append(self.tempX, np.array([[float(row[0]), float(row[1])]]), axis=0)
                tempY.append(float(row[2]))
            self.y = np.asarray(tempY)
                
    def generateOutput(self, output):
        with open(self.outStr, 'a') as csvFile:
            fileWriter = csv.writer(csvFile, lineterminator='\n')
            fileWriter.writerow(output)

    def scale(self):
        #Scale the data set using sklearn preprocessing scale function
        tempX2 = preprocessing.normalize(self.tempX, norm='l1')
        
        #create new matrix with constant
        for x in tempX2:
            self.x = np.append(self.x, np.array([[float(1), x[0], x[1]]]), axis=0)
        
        
                
def main():
    learning_Rates =[(0.001, 100), (0.005, 100), (0.01, 100), (0.05, 100), (0.1, 100)
                     , (0.5, 100), (1, 100), (5, 100), (10, 100), (0.2, 100)]
    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=str)
    parser.add_argument('o', type=str)

    args = parser.parse_args()

    value = inOut(args.i, args.o)

    for x in learning_Rates:
        SGD(value, x[0], x[1])


if __name__ == "__main__":
	main()
