# Unsupervised Learning: Introduction

#### Unsupervised Learning

无监督学习(Unsupervised Learning)可以分为两种：

- 化繁为简
    - 聚类(Clustering)
    - 降维(Dimension Reduction)
- 无中生有(Generation)

对于无监督学习(Unsupervised Learning)来说，我们通常只会拥有$(x,\hat y)$中的$x$或$\hat y$，其中：

- **化繁为简**就是把复杂的input变成比较简单的output，比如把一大堆没有打上label的树图片转变为一棵抽象的树，此时training data只有input $x$，而没有output $\hat y$
- **无中生有**就是随机给function一个数字，它就会生成不同的图像，此时training data没有input $x$，而只有output $\hat y$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/unsupervised.png" width="60%"/></center>

#### Clustering

##### Introduction

聚类，顾名思义，就是把相近的样本划分为同一类，比如对下面这些没有标签的image进行分类，手动打上cluster 1、cluster 2、cluster 3的标签，这个分类过程就是化繁为简的过程

一个很critical的问题：我们到底要分几个cluster？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/clustering.png" width="60%"/></center>

##### K-means

最常用的方法是**K-means**：

- 我们有一大堆的unlabeled data $\{x^1,...,x^n,...,x^N\}$，我们要把它划分为K个cluster
- 对每个cluster都要找一个center $c^i,i\in \{1,2,...,K\}$，initial的时候可以从training data里随机挑K个object $x^n$出来作为K个center $c^i$的初始值
- 遍历所有的object $x^n$，并判断它属于哪一个cluster，如果$x^n$与第i个cluster的center $c^i$最接近，那它就属于该cluster，我们用$b_i^n=1$来表示第n个object属于第i个cluster，$b_i^n=0$表示不属于
- 更新center：把每个cluster里的所有object取平均值作为新的center值，即$c^i=\sum\limits_{x^n}b_i^n x^n/\sum\limits_{x^n} b_i^n$
- 反复进行以上的操作

注：如果不是从原先的data set里取center的初始值，可能会导致部分cluster没有样本点

##### HAC

HAC，全称Hierarchical Agglomerative Clustering，层次聚类

假设现在我们有5个样本点，想要做clustering：

- build a tree:

    整个过程类似建立Huffman Tree，只不过Huffman是依据词频，而HAC是依据相似度建树

    - 对5个样本点两两计算相似度，挑出最相似的一对，比如样本点1和2
    - 将样本点1和2进行merge (可以对两个vector取平均)，生成代表这两个样本点的新结点
    - 此时只剩下4个结点，再重复上述步骤进行样本点的合并，直到只剩下一个root结点

- pick a threshold：

    选取阈值，形象来说就是在构造好的tree上横着切一刀，相连的叶结点属于同一个cluster

    下图中，不同颜色的横线和叶结点上不同颜色的方框对应着切法与cluster的分法

    <center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/HAC.png" width="60%"/></center>

HAC和K-means最大的区别在于如何决定cluster的数量，在K-means里，K的值是要你直接决定的；而在HAC里，你并不需要直接决定分多少cluster，而是去决定这一刀切在树的哪里

#### Dimension Reduction

##### Introduction

clustering的缺点是**以偏概全**，它强迫每个object都要属于某个cluster

但实际上某个object可能拥有多种属性，或者多个cluster的特征，如果把它强制归为某个cluster，就会失去很多信息；我们应该用一个vector来描述该object，这个vector的每一维都代表object的某种属性，这种做法就叫做Distributed Representation，或者说，Dimension Reduction

如果原先的object是high dimension的，比如image，那现在用它的属性来描述自身，就可以使之从高维空间转变为低维空间，这就是所谓的**降维(Dimension Reduction)**

下图为动漫“全职猎人”中小杰的念能力分布，从表中可以看出我们不能仅仅把他归为强化系

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/DR.png" width="60%"/></center>

##### Why Dimension Reduction Help?

接下来我们从另一个角度来看为什么Dimension Reduction可能是有用的

假设data为下图左侧中的3D螺旋式分布，你会发现用3D的空间来描述这些data其实是很浪费的，因为我们完全可以把这个卷摊平，此时只需要用2D的空间就可以描述这个3D的信息

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/DR2.png" width="60%"/></center>

如果以MNIST(手写数字集)为例，每一张image都是28\*28的dimension，但我们反过来想，大多数28\*28 dimension的vector转成image，看起来都不会像是一个数字，所以描述数字所需要的dimension可能远比28\*28要来得少

举一个极端的例子，下面这几张表示“3”的image，我们完全可以用中间这张image旋转$\theta$角度来描述，也就是说，我们只需要用$\theta$这一个dimension就可以描述原先28\*28 dimension的图像

你只要抓住角度的变化就可以知道28维空间中的变化，这里的28维pixel就是之前提到的樊一翁的胡子，而1维的角度则是他的头，也就是去芜存菁，化繁为简的思想

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/DR3.png" width="60%"/></center>

##### How to do Dimension Reduction？

在Dimension Reduction里，我们要找一个function，这个function的input是原始的x，output是经过降维之后的z

最简单的方法是**Feature Selection**，即直接从原有的dimension里拿掉一些直观上就对结果没有影响的dimension，就做到了降维，比如下图中从$x_1,x_2$两个维度中直接拿掉$x_1$；但这个方法不总是有用，因为很多情况下任何一个dimension其实都不能被拿掉，就像下图中的螺旋卷

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/DR4.png" width="60%"/></center>

另一个常见的方法叫做**PCA**(Principe Component Analysis)

PCA认为降维就是一个很简单的linear function，它的input x和output z之间是linear transform，即$z=Wx$，PCA要做的，就是根据一大堆的x**把W给找出来**(现在还不知道z长什么样子)

关于PCA算法的介绍详见下一篇文章



