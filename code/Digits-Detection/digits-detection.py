import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist


def load_data():  # categorical_crossentropy
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_test = np.random.normal(x_test)  # 加噪声
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    '''
    注意事项如下：
    1、batch_size=100,epochs=20为宜，batch_size过大会导致loss下降曲线过于平滑而卡在local minima、saddle point或plateau处，batch_size过小会导致update次数过多，运算量太大，速度缓慢，但可以带来一定程度的准确率提高
    2、hidden layer数量不要太多，不然可能会发生vanishing gradient(梯度消失)，一般两到三层为宜
    3、如果layer数量太多，则千万不要使用sigmoid等缩减input影响的激活函数，应当选择ReLU、Maxout等近似线性的activation function(layer数量不多也应该选这两个)
    4、每一个hidden layer所包含的neuron数量，五六百为宜
    5、对于分类问题，loss function一定要使用cross entropy(categorical_crossentropy)，而不是mean square error(mse)
    6、优化器optimizer一般选择adam，它综合了RMSProp和Momentum，同时考虑了过去的gradient、现在的gradient，以及上一次的惯性
    7、如果testing data上准确率很低，training data上准确率比较高，可以考虑使用dropout，Keras的使用方式是在每一层hidden layer的后面加上一句model.add(Dropout(0.5))，其中0.5这个参数你自己定；注意，加了dropout之后在training set上的准确率会降低，但是在testing set上的准确率会提高，这是正常的
    8、如果input是图片的pixel，注意对灰度值进行归一化，即除以255，使之处于0～1之间
    9、最后的output最好同时输出在training set和testing set上的准确率，以便于对症下药
    '''
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define network structure
    model = Sequential()

    model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=500, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='softmax'))

    # set configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    # evaluate the model and output the accuracy
    result_train = model.evaluate(x_train, y_train)
    result_test = model.evaluate(x_test, y_test)
    print('Train Acc:', result_train[1])
    print('Test Acc:', result_test[1])
