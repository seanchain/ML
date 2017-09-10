""" ~~ mnist_kaggle.py ~~
"""

# encoding: utf-8

from __future__ import print_function
import numpy as np

def convertTestData():
    lines = []
    with open('digit_test.csv') as f:
        lines = f.readlines()
    pixels = [] # pixels list
    lines = lines[1:len(lines)]
    for line in lines:
        content = line.split(',')
        pixel_l = []
        for i in range(0, 784):
            pixel_l.append(int(content[i].strip('\n')))
        pixels.append(pixel_l)
    pixels = np.asarray(pixels, dtype=float)
    print(pixels)
    np.save('x_test.npy', pixels)

def convertData():
    lines = []
    with open('digit_train.csv') as f:
        lines = f.readlines()
    labels = [] # labels list
    pixels = [] # pixels list
    lines = lines[1:len(lines)]
    for line in lines:
        content = line.split(',')
        labels.append(content[0])
        pixel_l = []
        for i in range(1, 785):
            pixel_l.append(int(content[i].strip('\n')))
        pixels.append(pixel_l)
    labels = np.asarray(labels, dtype=int)
    pixels = np.asarray(pixels, dtype=float)
    print(labels)
    print(pixels)
    np.save('x_train.npy', pixels)
    np.save('y_train.npy', labels)
    


def main():
    """ Main function
    """
    convertTestData()


if __name__ == '__main__':
    main()

