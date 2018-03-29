import sys
import argparse


def outputGenerate():
    f = open('output1.csv','w')
    f.write('w_1', 'w_2', 'b')
    f.close()
