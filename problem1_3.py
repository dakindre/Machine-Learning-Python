import sys
import argparse
import csv
import matplotlib.pyplot as plt



def plot(values):
    plt.scatter(values.xi, values.xj, c=values.labels)
    plt.show()
    
    
class inOut:
    def __init__(self, inStr, outStr, xi=[], xj=[], labels=[]):
        self.inStr = inStr
        self.outStr = outStr
        self.xi = xi
        self.xj = xj
        self.labels = labels
        self.generateInput()

    def generateInput(self):
        with open(self.inStr) as csvfile:
            inputCSV = csv.reader(csvfile, delimiter=',')
            for row in inputCSV:
                self.xi.append(float(row[0]))
                self.xj.append(float(row[1]))
                self.labels.append(float(row[2]))

    def generateOutput(self):
        return None


                 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=str)
    parser.add_argument('o', type=str)

    args = parser.parse_args()

    values = inOut(args.i, args.o)
    #print('xi ', values.xi)
    #print('xj ', values.xj)
    #print('labels ', values.labels)

    plot(values)
    

if __name__ == "__main__":
	main()
