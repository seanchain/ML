"""This file converts the original train and test set
to be some numpy file which is needed by us
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import math

def convert_sex(l):
    """Convert sex
    """
    new_l = []
    for item in l:
        if item == 'male':
            new_l.append(1)
        else:
            new_l.append(0)
    return new_l

def convert_sib_sp(l):
    """Convert sibling info
    """
    new_l = []
    for item in l:
        if item == 1 or item == 2:
            new_l.append(4)
        elif item == 0:
            new_l.append(3)
        elif item == 3:
            new_l.append(2)
        elif item == 4:
            new_l.append(1)
        else:
            new_l.append(0)
    return new_l

def convert_parch(l):
    """Convert Parent-Children info
    """
    new_l = []
    for item in l:
        if item == 1 or item == 2 or item == 3:
            new_l.append(3)
        elif item == 0:
            new_l.append(2)
        elif item == 5:
            new_l.append(1)
        else:
            new_l.append(0)
    return new_l

# def convert_embarked(l):
#     """Convert Embarked place to a single digit
#     """
#     new_l = []
#     for item in l:
#         if item == 'S':
#             new_l.append(1)
#         elif item == 'C':
#             new_l.append(2)
#         else:
#             new_l.append(3)
#     return new_l

def convert_age(l):
    new_l = []
    for item in l:
        if item >= 0 and item < 4:
            new_l.append(1) # baby
        elif item >= 0 and item < 13:
            new_l.append(2) # child
        elif item >= 13 and item < 18:
            new_l.append(3) # teen
        elif item >= 18 and item < 25:
            new_l.append(4) # young adult
        elif item >= 25 and item < 35:
            new_l.append(5) # adult
        elif item >= 35 and item < 60:
            new_l.append(6) # senior
        elif item >= 60 and item <= 160:
            new_l.append(7) # old
        else:
            new_l.append(0) # Not a number
    return new_l

def train_inputs():
    """Convert the original data to get train inputs
    """
    train_data = 'data/train.csv'
    df = pd.read_csv(train_data)
    pclass = df.Pclass.tolist()
    name = convert_name(df.Name.tolist())
    fare = convert_fare(df.Fare.tolist())
    cabin = convert_cabin(df.Cabin.tolist())
    sex = convert_sex(df.Sex.tolist())
    age = convert_age(df.Age.tolist())
    sib_sp = convert_sib_sp(df.SibSp.tolist())
    parch = convert_parch(df.Parch.tolist())
    train_num = len(pclass)
    total_l = []
    for i in range(train_num):
        l = []
        l.append(pclass[i])
        l.append(name[i])
        l.append(fare[i])
        l.append(cabin[i])
        l.append(sex[i])
        l.append(age[i])
        l.append(sib_sp[i])
        l.append(parch[i])
        total_l.append(l)
    res = np.asarray(total_l, dtype=float)
    return res

def convert_name(l):
    """Convert the name
    """
    new_l = []
    for item in l:
        name_prefix = item.split(', ')[1].split('.')[0].strip()
        if name_prefix == 'Mr':
            new_l.append(1)
        elif name_prefix == 'Dr':
            new_l.append(2)
        elif name_prefix == 'Col' or name_prefix == 'Major':
            new_l.append(3)
        elif name_prefix == 'Master':
            new_l.append(4)
        elif name_prefix == 'Miss':
            new_l.append(5)
        elif name_prefix == 'Mrs':
            new_l.append(6)
        elif name_prefix == 'Mme' or name_prefix == 'Ms' or name_prefix == 'Mlle' or name_prefix == 'Sir' or \
        name_prefix == 'Lady' or name_prefix == 'the Countness':
            new_l.append(7)
        else:
            new_l.append(0)
    return new_l

def convert_fare(l):
    """Convert the fare
    """
    new_l = []
    for item in l:
        if item >= 0 and item < 25:
            new_l.append(1)
        elif item >= 25 and item < 50:
            new_l.append(2)
        elif item >= 50 and item < 75:
            new_l.append(3)
        else:
            new_l.append(4)
    return new_l

def convert_cabin(l):
    """Convert the cabin
    """
    new_l = []
    for item in l:
        try:
            letter = item[0]
            if letter == 'A':
                new_l.append(1)
            elif letter == 'B':
                new_l.append(2)
            elif letter == 'C':
                new_l.append(3)
            elif letter == 'D':
                new_l.append(4)
            elif letter == 'E':
                new_l.append(5)
            elif letter == 'F':
                new_l.append(6)
            elif letter == 'G':
                new_l.append(7)
            else:
                new_l.append(0)
        except:
            new_l.append(0)
            continue
    return new_l

def test_inputs():
    """Convert the original data to get test inputs
    """
    test_data = 'data/test.csv'
    df = pd.read_csv(test_data)
    pclass = df.Pclass.tolist()
    name = convert_name(df.Name.tolist())
    fare = convert_fare(df.Fare.tolist())
    cabin = convert_cabin(df.Cabin.tolist())
    sex = convert_sex(df.Sex.tolist())
    age = convert_age(df.Age.tolist())
    sib_sp = convert_sib_sp(df.SibSp.tolist())
    parch = convert_parch(df.Parch.tolist())
    train_num = len(pclass)
    total_l = []
    for i in range(train_num):
        l = []
        l.append(pclass[i])
        l.append(name[i])
        l.append(fare[i])
        l.append(cabin[i])
        l.append(sex[i])
        l.append(age[i])
        l.append(sib_sp[i])
        l.append(parch[i])
        total_l.append(l)
    res = np.asarray(total_l, dtype=float)
    return res


def train_outputs():
    """Convert the original data to get train outputs
    """
    train_data = 'data/train.csv'
    df = pd.read_csv(train_data)
    survive = df.Survived.tolist()
    res = np.asarray(survive, dtype=float)
    return res

def fare():
    train_data = 'data/train.csv'
    df = pd.read_csv(train_data)
    fare = df.Fare.tolist()
    print(np.max(fare))

def main():
    """Main function
    """
    train_inputs()


if __name__ == "__main__":
    main()
