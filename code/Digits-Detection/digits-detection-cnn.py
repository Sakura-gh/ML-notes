import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

# categorical_crossentropy


def load_mnist_data(number):
    # the data, shuffled and  split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data(10000)

    # do DNN
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=100, epochs=20)

    result_train = model.evaluate(x_train, y_train)
    print('\nTrain Acc:\n', result_train[1])

    result_test = model.evaluate(x_test, y_test)
    print('\nTest Acc:\n', result_test[1])

    # do CNN
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    model2 = Sequential()
    model2.add(Conv2D(25, (3, 3), input_shape=(
        1, 28, 28), data_format='channels_first'))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Conv2D(50, (3, 3)))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dense(units=100, activation='relu'))
    model2.add(Dense(units=10, activation='softmax'))
    model2.summary()

    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

    model2.fit(x_train, y_train, batch_size=100, epochs=20)

    result_train = model2.evaluate(x_train, y_train)
    print('\nTrain CNN Acc:\n', result_train[1])
    result_test = model2.evaluate(x_test, y_test)
    print('\nTest CNN Acc:\n', result_test[1])
