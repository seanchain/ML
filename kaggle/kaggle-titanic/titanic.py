""" Run Titanic training process and predict on tests
"""
from __future__ import print_function

import titanic_data

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import pandas

def train_data():
    """ Train the data
    """
    batch_size = 1000
    num_classes = 2
    epochs = 5000

    x_train = titanic_data.train_inputs()
    y_train = titanic_data.train_outputs()

    # x_valid = titanic_data.train_inputs()
    # y_valid = titanic_data.train_outputs()

    print(len(x_train))

    x_train = x_train.reshape(891, 8)
    x_train = x_train.astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes)

    # x_valid = x_valid.reshape(241, 8)
    # x_valid = x_valid.astype('float32')
    # y_valid = keras.utils.to_categorical(y_valid, num_classes)

    model = Sequential()
    model.add(Dense(4, activation='sigmoid', input_shape=(8, )))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.003),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    new_input = titanic_data.test_inputs()
    new_input = new_input.astype('float32')
    outputs = model.predict(new_input)
    index_l = []
    for output in outputs:
        index = max(enumerate(output),key=lambda x: x[1])[0]
        index_l.append(index)
    df = pandas.read_csv('data/test.csv')
    passengers = df.PassengerId.tolist()
    output_csv_l = []
    for i in range(len(passengers)):
        output_csv_l.append([passengers[i], index_l[i]])
    with open("result.csv", "wb") as f:
        f.write(b'PassengerId,Survived\n')
        np.savetxt(f, np.asarray(output_csv_l).astype(int), fmt='%i', delimiter=",")


def main():
    train_data()
    

if __name__ == '__main__':
    main()
