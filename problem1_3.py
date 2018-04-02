import sys
import argparse
import csv
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


def PLA(value):
    '''Perceptron with feature values and lables'''
    clf = Perceptron(max_iter=100).fit(value.x, value.y)

    '''Attributes from Perceptron'''
    w_1 = clf.coef_[:, 0].astype(int)[0]
    w_2 = clf.coef_[:, 1].astype(int)[0]
    b = clf.intercept_.astype(int)[0]

    '''Output Perceptron attributes to csv file'''
    print(w_1, w_2, b)
    value.generateOutput(w_1, w_2, b)
    
    
    
class inOut:
    def __init__(self, inStr, outStr):
        self.inStr = inStr
        self.outStr = outStr
        self.x = np.array([[8,-1],[7,7],[12,-20],[14,-3],[12,8],[1,-12],[15,5],[7,-10],[10,4],[6,2],[8,12],[2,20],[1,-12],[9,8],[3,3],[5,6],[1,11]])
        self.y = np.array([1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1])
        self.generateInput()

    def generateInput(self):
        with open(self.inStr,) as csvfile:
            inputCSV = csv.reader(csvfile, delimiter=',')
            for row in inputCSV:
                None
                
    def generateOutput(self, w_1, w_2, b):
        rowString = [w_1, w_2, b]
        print (rowString)
        '''with open(self.outStr, 'a') as csvFile:
            fileWriter = csv.writer(csvFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            #fileWriter.writerow(w_1, w_2, b)'''
    
        

                 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=str)
    parser.add_argument('o', type=str)

    args = parser.parse_args()

    value = inOut(args.i, args.o)

    PLA(value)


    

if __name__ == "__main__":
	main()
