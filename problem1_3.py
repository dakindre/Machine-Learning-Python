import sys
import argparse
import csv
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt



'''def plot(values):
    plt.scatter(values.xi, values.xj, c=values.labels)
    plt.show()'''


def PLA(X, Y):
    h = .02
    clf = Perceptron(max_iter=100).fit(X, Y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax.axis('off')

    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

    ax.set_title('Perceptron')
    fig.show()
    
    
class inOut:
    def __init__(self, inStr, outStr):
        self.inStr = inStr
        self.outStr = outStr
        self.xi = np.array([[2,1],[3,4],[4,2],[3,1]])
        self.labels = np.array([0,0,1,1])
        self.generateInput()

    def generateInput(self):
        print('xi ', self.xi,' labels ', self.labels)
        with open(self.inStr) as csvfile:
            inputCSV = csv.reader(csvfile, delimiter=',')
            for row in inputCSV:
                None
                
    def generateOutput(self):
        return None

                 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=str)
    parser.add_argument('o', type=str)

    args = parser.parse_args()

    values = inOut(args.i, args.o)

    PLA(values.xi, values.labels)


    

if __name__ == "__main__":
	main()
