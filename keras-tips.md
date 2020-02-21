#### 实际调用toolkit-Keras去训练model时的注意事项如下：

##### 搭建神经网络架构

1、batch_size=100,epochs=20为宜，batch_size过大会导致loss下降曲线过于平滑而卡在local minima、saddle point或plateau处，batch_size过小会导致update次数过多，运算量太大，速度缓慢，但可以带来一定程度的准确率提高

2、hidden layer数量不要太多，不然可能会发生vanishing gradient(梯度消失)，一般两到三层为宜

3、如果layer数量太多，则千万不要使用sigmoid等缩减input影响的激活函数，应当选择ReLU、Maxout等近似线性的activation function(layer数量不多也应该选这两个)

4、每一个hidden layer所包含的neuron数量，五六百为宜

5、对于分类问题，loss function一定要使用cross entropy(categorical_crossentropy)，而不是mean square error(mse)

6、优化器optimizer一般选择adam，它综合了RMSProp和Momentum，同时考虑了过去的gradient、现在的gradient，以及上一次的惯性

7、如果testing data上准确率很低，training data上准确率比较高，可以考虑使用dropout，Keras的使用方式是在每一层hidden layer的后面加上一句model.add(Dropout(0.5))，其中0.5这个参数你自己定；注意，加了dropout之后在training set上的准确率会降低，但是在testing set上的准确率会提高，这是正常的

8、如果input是图片的pixel，注意对灰度值进行归一化，即除以255，使之处于0～1之间

9、最后的output最好同时输出在training set和testing set上的准确率，以便于对症下药

##### 针对training data上的performance

训练神经网络的第一要义是，**一定要先把training data上的performance提高上去**

1、Activation function使用ReLU或Maxout

2、Adaptive learning rate使用Adam

3、如果上面两条都未能提高training set上的准确率，这说明你的network还没有能力可以fit training data，尝试更改network structure，包括layer的层数和每层layer中neuron的个数

##### 针对testing data上的performance：

1、如果training set上performance比较差，不管testing set表现怎么样，都回过头去提升training data上的performance

2、如果training set上performance比较好，而testing set上performance比较差，说明发生了overfitting，采用dropout的方式

