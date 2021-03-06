import sys
import argparse
import csv
import numpy as np
from sklearn.linear_model import Perceptron


def PLA(value):
    '''Perceptron with feature values and lables'''
    clf = Perceptron(n_iter=100).fit(value.x, value.y)
    

    '''Attributes from Perceptron'''
    w_1 = clf.coef_[:, 0].astype(int)[0]
    w_2 = clf.coef_[:, 1].astype(int)[0]
    b = clf.intercept_.astype(int)[0]

    '''Output Perceptron attributes to csv file'''
    wL = [w_1,w_2,b]
    value.generateOutput(wL)
    
    
    
class inOut:
    def __init__(self, inStr, outStr):
        self.inStr = inStr
        self.outStr = outStr
        self.x = np.empty((0,2), int)
        self.y = np.empty((0,1), int)
        self.generateInput()

    def generateInput(self):
        tempList = []
        with open(self.inStr) as csvfile:
            inputCSV = csv.reader(csvfile, delimiter=',')
            for row in inputCSV:
                self.x = np.append(self.x, np.array([[row[0], row[1]]]), axis=0)
                tempList.append(int(row[2]))
            self.y = np.asarray(tempList)
                
    def generateOutput(self, wL):
        with open(self.outStr, 'a') as csvFile:
            fileWriter = csv.writer(csvFile, lineterminator='\n')
            fileWriter.writerow(wL)
            
                 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=str)
    parser.add_argument('o', type=str)

    args = parser.parse_args()

    value = inOut(args.i, args.o)

    PLA(value)


    
if __name__ == "__main__":
	main()
