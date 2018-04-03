import sys
import argparse
import csv
import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


def LR(value):
    print(None)
    
    
    
    
    
class inOut:
    def __init__(self, inStr, outStr):
        self.inStr = inStr
        self.outStr = outStr
        self.x = np.empty((0,3), int)
        #self.y = np.empty([])
        self.B = np.zeros(4)
        self.generateInput()

    def generateInput(self):
        with open(self.inStr) as csvfile:
            inputCSV = csv.reader(csvfile, delimiter=',')
            for row in inputCSV:
                self.x = np.append(self.x, np.array([[row[0], row[1], row[2]]]), axis=0)
            print (self.x)
                
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

    LR(value)


    

if __name__ == "__main__":
	main()
