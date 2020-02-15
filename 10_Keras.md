# Keras2.0

#### Why Keras

你可能会问，为什么不学TensorFlow呢？明明tensorflow才是目前最流行的machine learning库之一啊。其实，它并没有那么好用，tensorflow和另外一个功能相近的toolkit theano，它们是非常flexible的，你甚至可以把它想成是一个微分器，它完全可以做deep learning以外的事情，因为它的作用就是帮你算微分，拿到微分之后呢，你就可以去算gradient descent之类，而这么flexible的toolkit学起来是有一定的难度的，你没有办法在半个小时之内精通这个toolkit

但是另一个toolkit——Keras，你是可以在数十分钟内就熟悉并精通它的，然后用它来implement一个自己的deep learning，Keras其实是tensorflow和theano的interface，所以用Keras就等于在用tensorflow，只是有人帮你把操纵tensorflow这件事情先帮你写好

所以Keras是比较容易去学习和使用的，并且它也有足够的弹性，除非你自己想要做deep learning的研究，去设计一个自己的network，否则多数你可以想到的network，在Keras里都有现成的function可以拿来使用；因为它背后就是tensorflow or theano，所以如果你想要精进自己的能力的话，你永远可以去改Keras背后的tensorflow的code，然后做更厉害的事情

而且，现在Keras已经成为了Tensorflow官方的API，它像搭积木一样简单

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras.png" width="50%;" /></center>
接下来我们用手写数字识别的demo来介绍一下"Hello world" of deep learning

#### prepare data

使用的data是MNIST的Data：http://yann.lecun.com/exdb/mnist/

Keras提供了自动下载MNIST data的function：http://keras.io/datasets/

#### process

首先要先导入keras包：`from keras.models import Sequential`

##### step 1：define a set of function——neural network

先用`Sequential()`宣告建立一个model

~~~python
model = Sequential()
~~~

然后开始叠一个neural network：它有两个hidden layer，每个hidden layer都有500个neuron

- 加一个**Fully connected**的layer——用**Dense**来表示，当然你也可以加别的layer，比如convolution的layer

    之前我们说过，input layer比较特殊，它并不是真正意义上的layer，因为它没有所谓的"neuron"，于是Keras在model里面加的第一层layer会有一些特殊，要求同时输入`input_dim`和`units`，分别代表第一层hidden layer输入维数(也就是input layer的dimension)和第一层hidden layer的neuron个数

    `input_dim=28*28`表示一个28*28=784长度的vector，代表image；`units=500`表示该层hidden layer要有500个neuron；`activation=‘sigmoid’`表示激活函数使用sigmoid function

    ~~~python
    model.add(Dense(input_dim=28 * 28, units=500, activation='sigmoid'))
    ~~~

    加完layer之后，还需要设定该层hidden layer所使用的activation function，这里直接就用sigmoid function

    在Keras里还可以选别的activation function，比如softplus、softsign、relu、tanh、hard_sigmoid、linear等等，如果你要加上自己的activation function，其实也蛮容易的，只要在Keras里面找到写activation function的地方，自己再加一个进去就好了

- 从第二层hidden layer开始，如果要在model里再加一个layer，就用model.add增加一个Dense全连接层，包括`units`和`activation`参数

    这边就不需要再redefine `input_dim`是多少了，因为新增layer的input就等于前一个layer的output，Keras自己是知道这件事情的，所以你就直接告诉它说，新加的layer有500个neuron就好了

    这里同样把activation function设置为sigmoid function

    ~~~python
    model.add(Dense(units=500, activation='sigmoid'))
    ~~~

- 最后，由于是分10个数字，所以output是10维，如果把output layer当做一个Multi-class classifier的话，那activation function就用softmax(这样可以让output每一维的几率之和为1，表现得更像一个概率分布)，当然你也可以选择别的

    ~~~python
    model.add(Dense(units=10, activation='softmax'))
    ~~~

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-step1.png" width="60%;" /></center>
注：上图中写的是Keras1.0的语法，在笔记中给出的则是Keras2.0的语法，应当使用后者

##### Step 2：goodness of function——cross entropy

evaluate一个function的好坏，你要做的事情是用model.compile去定义你的loss function是什么

比如说你要用**cross entropy**的话，那你的loss参数就是**categorical_crossentropy**(Keras里的写法)

~~~python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
~~~

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-step2.png" width="60%;" /></center>
##### Step 3：pick the best function

###### Configuration

在training之前，你要先下一些**configuration**告诉它training的时候，你打算要怎么做

你要定义的第一个东西是optimizer，也就是说，你要用什么样的方式来找最好的function，虽然optimizer后面可以接不同的方式，但是这些不同的方式，其实都是gradient descent类似的方法

有一些方法machine会自动地，empirically(根据经验地)决定learning rate的值应该是多少，所以这些方法是不需要给它learning rate的，Keras里面有诸如：SGD(gradient descent)、RMSprop、Adagrad、Adadelta、Adam、Adamax、Nadam之类的寻找最优参数的方法，它们都是gradient descent的方式

~~~python
model.compile(loss='categorical crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
~~~

###### Training

决定好怎么做gradient descent之后，就是实际去做训练了，去跑gradient descent找最优参数了

这里使用的是`model.fit`方法，要给它4给input(假设我们给了10000张image作Training data)

- 第一个input是Training data——`x_train`

    在这个case里，Training data就是一张一张的image，需要把它存放到numpy array里面，这个numpy array是two-dimension的matrix，每张image存为numpy array的一个行向量(它把image中28\*28个像素值拉成一个行向量)，总共有10000行，它的列数就是每张image的像素点个数，即28\*28=784列

- 第二个input是每一个Training data对应的label——`y_train`

    在这个case里，就是标志着这张image对应的是0~9的那一个数字，同样也是two-dimension的numpy array，每张image的label存为numpy array的一个行向量，用来表示0~9这10个数字中的某一个数，所以是10列，用的是one-hot编码，10个数字中对了对应image的那个数字为1之外其余都是0

- 第三个input是`batch_size`，告诉Keras我们的batch要有多大

    在这个case里，batch_size=100，表示我们要把100张随机选择的image放到一个batch里面，然后把所有的image分成一个个不同的batch，Keras会自动帮你完成随机选择image的过程，不需要自己去code

- 第四个input是`nb_epoch`，表示对所有batch的训练要做多少次

    在这个case里，nb_epoch=20，表示要对所有的batch进行20遍gradient descent的训练，每看到一个batch就update一次参赛，假设现在每一个epoch里面有100个batch，就对应着update 100次参数，20个epoch就是update 2000次参数

注：如果batch_size设为1，就是**Stochastic Gradient Descent**(随机梯度下降法)，这个我们之前在讨论gradient descent的时候有提到，就是每次拿到一个样本点就update一次参数，而不是每次拿到一批样本点的error之后才去update参数，因此stochastic gradient descent的好处是它的速度比较快，虽然每次update参数的方向是不稳定的，但是**天下武功，唯快不破**，在别人出一拳的时候，它就已经出了100拳了，所以它是比较强的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-step3.png" width="60%;" /></center>
#### Mini-batch

这里有一个秘密，就是我们在做deep learning的gradient descent的时候，并不会真的去minimize total loss，那我们做的是什么呢？我们会把Training data分成一个一个的batch，比如说你的Training data一共有1w张image，每次random选100张image作为一个batch(我的理解是，先将原来的image分布随机打乱，然后再按顺序每次挑出batch_size张image组成一个batch，这样才能保证所有的data都有被用到，且不同的batch里不会出现重复的data)

- 像gradient descent一样，先随机initialize network的参数

- 选第一个batch出来，然后计算这个batch里面的所有element的total loss，$L'=l^1+l^{31}+...$，接下来根据$L'$去update参数，也就是计算$L'$对所有参数的偏微分，然后update参数

    注意：不是全部data的total loss

- 再选择第二个batch，现在这个batch的total loss是$L''=l^2+l^{16}+...$，接下来计算$L''$对所有参数的偏微分，然后update参数

- 反复做这个process，直到把所有的batch通通选过一次，所以假设你有100个batch的话，你就把这个参数update 100次，把所有batch看过一次，就叫做一个epoch

- 重复epoch的过程，所以你在train network的时候，你会需要好几十个epoch，而不是只有一个epoch

整个训练的过程类似于stochastic gradient descent，不是将所有数据读完才开始做gradient descent的，而是拿到一部分数据就做一次gradient descent

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/mini-batch.png" width="50%;" /></center>
#### Batch size and Training Speed

##### batch size太小会导致不稳定，速度上也没有优势

前面已经提到了，stochastic gradient descent速度快，表现好，既然如此，为什么我们还要用Mini-batch呢？这就涉及到了一些实际操作上的问题，让我们必须去用Mini-batch

举例来说，我们现在有50000个examples，如果我们把batch size设置为1，就是stochastic gradient descent，那在一个epoch里面，就会update 50000次参数；如果我们把batch size设置为10，在一个epoch里面，就会update 5000次参数

看上去stochastic gradient descent的速度貌似是比较快的，它一个epoch更新参数的次数比batch size等于10的情况下要快了10倍，但是！我们好像忽略了一个问题，我们之前一直都是下意识地认为不同batch size的情况下运行一个epoch的时间应该是相等的，然后我们才去比较每个epoch所能够update参数的次数，可是它们又怎么可能会是相等的呢？

实际上，当你batch size设置不一样的时候，一个epoch需要的时间是不一样的，以GTX 980为例，下图是对总数为50000笔的Training data设置不同的batch size时，每一个epoch所需要花费的时间

- case1：如果batch size设为1，也就是stochastic gradient descent，一个epoch要花费166秒，接近3分钟

- case2：如果batch size设为10，那一个epoch是17秒

也就是说，当stochastic gradient descent算了一个epoch的时候，batch size为10的情况已经算了近10个epoch了；所以case1跑一个epoch，做了50000次update参数的同时，case2跑了十个epoch，做了近5000\*10=50000次update参数；你会发现batch size设1和设10，update参数的次数几乎是一样的

如果不同batch size的情况，update参数的次数几乎是一样的，你其实会想要选batch size更大的情况，就像在本例中，相较于batch size=1，你会更倾向于选batch size=10，因为batch size=10的时候，是会比较稳定的，因为**由更大的数据集计算的梯度能够更好的代表样本总体，从而更准确的朝向极值所在的方向**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/batch-size-speed.png" width="50%;" /></center>
我们之前把gradient descent换成stochastic gradient descent，是因为后者速度比较快，update次数比较多，可是现在如果你用stochastic gradient descent并没有见得有多快，那你为什么不选一个update次数差不多，又比较稳定的方法呢？

##### batch size会受到GPU平行加速的限制，太大可能导致在train的时候卡住

上面例子的现象产生的原因是我们用了GPU，用了平行运算，所以batch size=10的时候，这10个example其实是同时运算的，所以你在一个batch里算10个example的时间跟算1个example的时间几乎可以是一样的

那你可能会问，既然batch size越大，它会越稳定，而且还可以平行运算，那为什么不把batch size变得超级大呢？这里有两个claim(声明)：

- 第一个claim就是，如果你把batch size开到很大，最终GPU会没有办法进行平行运算，它终究是有自己的极限的，也就是说它同时考虑10个example和1个example的时间是一样的，但当它考虑10000个example的时候，时间就不可能还是跟一个example一样，因为batch size考虑到**硬件限制**，是没有办法无穷尽地增长的

- 第二个claim是说，如果把batch size设的很大，在train gradient descent的时候，可能跑两下你的network就卡住了，就陷到saddle point或者local minima里面去了

    因为在neural network的error surface上面，如果你把loss的图像可视化出来的话，它并不是一个convex的optimization problem，不会像理想中那么平滑，实际上它会有很多的坑坑洞洞

    如果你用的batch size很大，甚至是Full batch，那你走过的路径会是比较平滑连续的，可能这一条平滑的曲线在走向最低点的过程中就会在坑洞或是缓坡上卡住了；但是，如果你的batch size没有那么大，意味着你走的路线没有那么的平滑，有些步伐走的是**随机性**的，路径是会有一些曲折和波动的

    可能在你走的过程中，它的曲折和波动刚好使得你“绕过”了那些saddle point或是local minima的地方；或者当你陷入不是很深的local minima或者没有遇到特别麻烦的saddle point的时候，它步伐的随机性就可以帮你跳出这个gradient接近于0的区域，于是你更有可能真的走向global minima的地方

    而对于Full batch的情况，它的路径是没有随机性的，是稳定朝着目标下降的，因此在这个时候去train neural network其实是有问题的，可能update两三次参数就会卡住，所以mini batch是有必要的

    下面是我手画的图例和注释：

    <center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/batch-size.jpg" width="70%;" /></center>

##### 不同batch size在梯度下降上的表现

如下图，左边是full batch(拿全部的Training data做一个batch)的梯度下降效果，可以看到每一次迭代成本函数都呈现下降趋势，这是好的现象，说明我们w和b的设定一直再减少误差， 这样一直迭代下去我们就可以找到最优解；右边是mini batch的梯度下降效果，可以看到它是上下波动的，成本函数的值有时高有时低，但总体还是呈现下降的趋势， 这个也是正常的，因为我们每一次梯度下降都是在min batch上跑的而不是在整个数据集上， 数据的差异可能会导致这样的波动(可能某段数据效果特别好，某段数据效果不好)，但没关系，因为它整体是呈下降趋势的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-gd1.png" width="50%;" /></center>
把下面的图看做是梯度下降空间：蓝色部分是full batch而紫色部分是mini batch，就像上面所说的mini batch不是每次迭代损失函数都会减少，所以看上去好像走了很多弯路，不过整体还是朝着最优解迭代的，而且由于mini batch一个epoch就走了5000步(5000次梯度下降)，而full batch一个epoch只有一步，所以虽然mini batch走了弯路但还是会快很多

而且，就像之前提到的那样，mini batch在update的过程中，步伐具有随机性，因此紫色的路径可以在一定程度上绕过或跳出saddle point、local minima这些gradient趋近于0的地方；而蓝色的路径因为缺乏随机性，只能按照既定的方式朝着目标前进，很有可能就在中途被卡住，永远也跳不出来了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-gd2.png" width="40%;" /></center>
当然，就像之前讨论的一样，如果batch size太小，会造成速度不仅没有加快反而会导致下降的曲线更加不稳定的情况产生

==**因此batch size既不能太大，因为它会受到硬件GPU平行加速的限制，导致update次数过于缓慢，并且由于缺少随机性而很容易在梯度下降的过程中卡在saddle point或是local minima的地方(极端情况是Full batch)；而且batch size也不能太小，因为它会导致速度优势不明显的情况下，梯度下降曲线过于不稳定，算法可能永远也不会收敛(极端情况是Stochastic gradient descent)**==

##### GPU是如何平行加速的

整个network，不管是Forward pass还是Backward pass，都可以看做是一连串的矩阵运算的结果

那今天我们就可以比较batch size等于1(stochastic gradient descent)和10(mini batch)的差别

如下图所示，stochastic gradient descent就是对每一个input x进行单独运算；而mini batch，则是把同一个batch里面的input全部集合起来，假设现在我们的batch size是2，那mini batch每一次运算的input就是把黄色的vector和绿色的vector拼接起来变成一个matrix，再把这个matrix乘上$w_1$，你就可以直接得到$z^1$和$z^2$

这两件事在理论上运算量是一样多的，但是在实际操作上，对GPU来说，在矩阵里面相乘的每一个element都是可以平行运算的，所以图中stochastic gradient descent运算的时间反而会变成下面mini batch使用GPU运算速度的两倍，这就是为什么我们要使用mini batch的原因

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/matrix-speed.png" width="50%;" /></center>
所以，如果你买了GPU，但是没有使用mini batch的话，其实就不会有多少加速的效果

#### Save and Load Models

Keras是可以帮你save和load model的，你可以把train好的model存起来，以后再用另外一个程式读出来，它也可以帮你做testing

那怎么用neural network去testing呢？有两种case：

- case 1是**evaluation**，比如今天我有一组testing set，testing set的答案也是已知的，那Keras就可以帮你算现在的正确率有多少，这个`model.evaluate`函数有两个input，就是testing的image和testing的label

    ~~~python
    score = model.evaluate(x_test,y_test)
    print('Total loss on Testing Set:',score[0])
    print('Accuracy of Testing Set:',score[1])
    ~~~

- case 2是**prediction**，这个时候`model.predict`函数的input只有image data而没有任何的label data，output就直接是分类的结果

    ~~~python
    result = model.predict(x_test)
    ~~~

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/save-load-model.png" width="60%;" /></center>
#### Appendix：手写数字识别完整代码(Keras2.0)

##### code

~~~python
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

# categorical_crossentropy
def load_data():  
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
    # x_test=np.random.normal(x_test)
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define network structure
    model = Sequential()

    model.add(Dense(input_dim=28*28, units=500, activation='sigmoid'))
    model.add(Dense(units=500, activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))

    # set configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    # evaluate the model and output the accuracy
    result = model.evaluate(x_test, y_test)
    print('Test Acc:', result[1])

~~~

##### result

~~~python
Epoch 1/20
10000/10000 [==============================] - 2s 214us/step - loss: 1.1724 - acc: 0.6558
Epoch 2/20
10000/10000 [==============================] - 1s 146us/step - loss: 0.3847 - acc: 0.8964
Epoch 3/20
10000/10000 [==============================] - 1s 132us/step - loss: 0.2968 - acc: 0.9119
Epoch 4/20
10000/10000 [==============================] - 1s 146us/step - loss: 0.2535 - acc: 0.9268
Epoch 5/20
10000/10000 [==============================] - 2s 185us/step - loss: 0.2284 - acc: 0.9332
Epoch 6/20
10000/10000 [==============================] - 1s 141us/step - loss: 0.2080 - acc: 0.9369
Epoch 7/20
10000/10000 [==============================] - 1s 135us/step - loss: 0.1829 - acc: 0.9455
Epoch 8/20
10000/10000 [==============================] - 1s 135us/step - loss: 0.1617 - acc: 0.9520
Epoch 9/20
10000/10000 [==============================] - 1s 136us/step - loss: 0.1470 - acc: 0.9563
Epoch 10/20
10000/10000 [==============================] - 1s 133us/step - loss: 0.1340 - acc: 0.9607
Epoch 11/20
10000/10000 [==============================] - 1s 141us/step - loss: 0.1189 - acc: 0.9651
Epoch 12/20
10000/10000 [==============================] - 1s 143us/step - loss: 0.1056 - acc: 0.9696
Epoch 13/20
10000/10000 [==============================] - 1s 140us/step - loss: 0.0944 - acc: 0.9728
Epoch 14/20
10000/10000 [==============================] - 2s 172us/step - loss: 0.0808 - acc: 0.9773
Epoch 15/20
10000/10000 [==============================] - 1s 145us/step - loss: 0.0750 - acc: 0.9800
Epoch 16/20
10000/10000 [==============================] - 1s 134us/step - loss: 0.0643 - acc: 0.9826
Epoch 17/20
10000/10000 [==============================] - 1s 132us/step - loss: 0.0568 - acc: 0.9850
Epoch 18/20
10000/10000 [==============================] - 1s 135us/step - loss: 0.0510 - acc: 0.9873
Epoch 19/20
10000/10000 [==============================] - 1s 134us/step - loss: 0.0434 - acc: 0.9898
Epoch 20/20
10000/10000 [==============================] - 1s 134us/step - loss: 0.0398 - acc: 0.9906
10000/10000 [==============================] - 1s 79us/step
Test Acc: 0.9439
~~~

可以发现每次做完一个epoch的update后，手写数字识别的准确率都有上升，最终训练好的model识别准确率等于94.39%

注：把activation function从sigmoid换成relu可以使识别准确率更高，这里不再重复试验