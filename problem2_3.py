import sys
import argparse
import csv
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model


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
