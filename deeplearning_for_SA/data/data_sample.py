### this is to test
"""
from __future__ import print_function

filename = "abcxyz"
print('Succesfully downloaded', filename, 'bytes.')


def read_labels(filename, one_hot = False):
    print('Extracting', filename)
    for line in open(filename, "r"):
        line = line.strip()
        labels = line.from
"""
import numpy as np
from io import StringIO

#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
"""
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print args.accumulate(args.integers)

"""

# import tensorflow as tf
# import numpy as np
if __name__ == "__main__":
    """
    a = np.random.rand(3,3,4)
    print a
    print "======="
    tem = np.copy(a[0][0])
    a[0][0] = np.copy(a[0][2])
    a[0][2] = np.copy(tem)
    print a
    print tem
    """

    a = [1, 2, 2, 3, 4, 5, 6, 7, 8, 9]

    b = a[-4:]
    print b
    #source = "Dictionary_list_fixed"

    #c = StringIO(u"1 2/n 3 4")


    #data = np.loadtxt(source, dtype={'names':('word', 'class'),'formats':('S10', 'i4')})
    #data = np.genfromtxt(source, delimiter= " ")
    #data
    #print (unicode(data[1]).encode("utf8"))
    """
    a = np.arange(12)
    print (a)
    b = a.reshape(2,2,3,1)
    print (b)
    print ("start")
    abcxyz = np.empty(3)
    for x in range(3):
        abcxyz[x] = x
    #abcxyz = np.append([1,2,3],axis = 0)

    a = np.zeros(10)
    print (a+1)
    a[2] = 1.2
    print (a)
    print (a.dtype)


    for pth in sys.path:
        print pth

    s_label = [0] * 10
    y = "9"
    s_label[int(y)] = 1
    print (s_label)
"""


