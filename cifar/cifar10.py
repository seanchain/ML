#! /usr/bin/python

''' ~~ cifar.py ~~
Try the caffe quick method for cifar-10
problem
'''

from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm

def train_and_evaluate():
    """ Training and Testing method
    """
    batch_size = 32
    num_classes = 10
    epochs = 50

    img_rows = 32
    img_cols = 32    
    x_train = np.load('data/x_train.npy')
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3).astype('float32')
    x_test = np.load('data/x_test.npy')
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3).astype('float32')
    
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255
    
    print(x_train[0])

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    checkpoint = ModelCheckpoint(filepath='best-model.hdf5', monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max')
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[checkpoint])
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss: ", score[0])
    print("Test Acc: ", score[1])
    

def main():
    train_and_evaluate()

if __name__ == '__main__':
    main()
