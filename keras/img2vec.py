
from __future__ import print_function

import scipy.misc
import numpy as np

def getTests():
    ary = scipy.misc.imread('0.png', flatten=True)
    l = []
    for val in ary:
        for number in val:
            l.append(255 - number)

    ary1 = scipy.misc.imread('1.png', flatten=True)
    l1 = []
    for val1 in ary1:
        for number1 in val1:
            l1.append(255 - number1)
    ary2 = scipy.misc.imread('2.png', flatten=True)
    l2 = []
    for val2 in ary2:
        for number2 in val2:
            l2.append(255 - number2)
    ary3 = scipy.misc.imread('3.png', flatten=True)
    l3 = []
    for val3 in ary3:
        for number3 in val3:
            l3.append(255 - number3)
    ary4 = scipy.misc.imread('4.png', flatten=True)
    l4 = []
    for val4 in ary4:
        for number4 in val4:
            l4.append(255 - number4)
    
    ary5 = scipy.misc.imread('5.png', flatten=True)
    l5 = []
    for val5 in ary5:
        for number5 in val5:
            l5.append(255 - number5)

    new_l = []
    new_l.append(l)
    new_l.append(l1)
    new_l.append(l2)
    new_l.append(l3)
    new_l.append(l4)
    new_l.append(l5)

    new_ary = np.asarray(new_l, dtype=float)
    return new_ary