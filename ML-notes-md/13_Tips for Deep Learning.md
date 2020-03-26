# Tips for Deep Learning

> 本文会顺带解决CNN部分的两个问题：
> 1、max pooling架构中用到的max无法微分，那在gradient descent的时候该如何处理？
> 2、L1 的Regression到底是什么东西
>
> 本文的主要思路：针对training set和testing set上的performance分别提出针对性的解决方法
> 1、在training set上准确率不高：
> 	  new activation function：ReLU、Maxout
> 	  adaptive learning rate：Adagrad、RMSProp、Momentum、Adam
> 2、在testing set上准确率不高：Early Stopping、Regularization or Dropout

### Recipe of Deep Learning

#### three step of deep learning

Recipe，配方、秘诀，这里指的是做deep learning的流程应该是什么样子

我们都已经知道了deep learning的三个步骤

- define the function set(network structure) 
- goodness of function(loss function -- cross entropy)
- pick the best function(gradient descent -- optimization)

做完这些事情以后，你会得到一个更好的neural network，那接下来你要做什么事情呢？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/recipe-dl.png" width="60%;" /></center>
#### Good Results on Training Data？

你要做的第一件事是，**提高model在training set上的正确率**

先检查training set的performance其实是deep learning一个非常unique的地方，如果今天你用的是k-nearest neighbor或decision tree这类非deep learning的方法，做完以后你其实会不太想检查training set的结果，因为在training set上的performance正确率就是100，没有什么好检查的

有人说deep learning的model里这么多参数，感觉一脸很容易overfitting的样子，但实际上这个deep learning的方法，它才不容易overfitting，我们说的**overfitting就是在training set上performance很好，但在testing set上performance没有那么好**；只有像k nearest neighbor，decision tree这类方法，它们在training set上正确率都是100，这才是非常容易overfitting的，而对deep learning来说，overfitting往往不会是你遇到的第一个问题

因为你在training的时候，deep learning并不是像k nearest neighbor这种方法一样，一训练就可以得到非常好的正确率，它有可能在training set上根本没有办法给你一个好的正确率，所以，这个时候你要回头去检查在前面的step里面要做什么样的修改，好让你在training set上可以得到比较高的正确率

#### Good Results on Testing Data？

接下来你要做的事是，**提高model在testing set上的正确率**

假设现在你已经在training set上得到好的performance了，那接下来就把model apply到testing set上，我们最后真正关心的，是testing set上的performance，假如得到的结果不好，这个情况下发生的才是Overfitting，也就是在training set上得到好的结果，却在testing set上得到不好的结果

那你要回过头去做一些事情，试着解决overfitting，但有时候你加了新的technique，想要overcome overfitting这个problem的时候，其实反而会让training set上的结果变坏；所以你在做完这一步的修改以后，要先回头去检查新的model在training set上的结果，如果这个结果变坏的话，你就要从头对network training的process做一些调整，那如果你同时在training set还有testing set上都得到好结果的话，你就成功了，最后就可以把你的系统真正用在application上面了

#### Do not always blame overfitting

不要看到所有不好的performance就归责于overfitting

先看右边testing data的图，横坐标是model做gradient descent所update的次数，纵坐标则是error rate(越低说明model表现得越好)，黄线表示的是20层的neural network，红色表示56层的neural network

你会发现，这个56层network的error rate比较高，它的performance比较差，而20层network的performance则是比较好的，有些人看到这个图，就会马上得到一个结论：56层的network参数太多了，56层果然没有必要，这个是overfitting。但是，真的是这样子吗？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/blame-over.png" width="60%;" /></center>
你在说结果是overfitting之前，有检查过training set上的performance吗？对neural network来说，在training set上得到的结果很可能会像左边training error的图，也就是说，20层的network本来就要比56层的network表现得更好，所以testing set得到的结果并不能说明56层的case就是发生了overfitting

在做neural network training的时候，有太多太多的问题可以让你的training set表现的不好，比如说我们有local minimum的问题，有saddle point的问题，有plateau的问题...所以这个56层的neural network，有可能在train的时候就卡在了一个local minimum的地方，于是得到了一个差的参数，但这并不是overfitting，而是在training的时候就没有train好

有人认为这个问题叫做underfitting，但我的理解上，**underfitting**的本意应该是指这个model的complexity不足，这个model的参数不够多，所以它的能力不足以解出这个问题；但这个56层的network，它的参数是比20层的network要来得多的，所以它明明有能力比20层的network要做的更好，却没有得到理想的结果，这种情况不应该被称为underfitting，其实就只是没有train好而已

#### conclusion

当你在deep learning的文献上看到某种方法的时候，永远要想一下，这个方法是要解决什么样的问题，因为在deep learning里面，有两个问题：

- 在training set上的performance不够好
- 在testing set上的performance不够好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/different-methods.png" width="60%;" /></center>
当只有一个方法propose(提出)的时候，它往往只针对这两个问题的其中一个来做处理，举例来说，deep learning有一个很潮的方法叫做dropout，那很多人就会说，哦，这么潮的方法，所以今天只要看到performance不好，我就去用dropout；但是，其实只有在testing的结果不好的时候，才可以去apply dropout，如果你今天的问题只是training的结果不好，那你去apply dropout，只会越train越差而已

所以，你**必须要先想清楚现在的问题到底是什么，然后再根据这个问题去找针对性的方法**，而不是病急乱投医，甚至是盲目诊断

下面我们分别从Training data和Testing data两个问题出发，来讲述一些针对性优化的方法

### Good Results on Training Data？

这一部分主要讲述如何在Training data上得到更好的performance，分为两个模块，New activation function和Adaptive Learning Rate

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/different-method.png" width="60%;" /></center>
#### New activation function

这个部分主要讲述的是关于Recipe of Deep Learning中New activation function的一些理论

##### activation function

如果你今天的training结果不好，很有可能是因为你的network架构设计得不好。举例来说，可能你用的activation function是对training比较不利的，那你就尝试着换一些新的activation function，也许可以带来比较好的结果

在1980年代，比较常用的activation function是sigmoid function，如果现在我们使用sigmoid function，你会发现deeper不一定imply better，下图是在MNIST手写数字识别上的结果，当layer越来越多的时候，accuracy一开始持平，后来就掉下去了，在layer是9层、10层的时候，整个结果就崩溃了；但注意！9层、10层的情况并不能被认为是因为参数太多而导致overfitting，实际上这张图就只是training set的结果，你都不知道testing的情况，又哪来的overfitting之说呢？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-not-ok.png" width="60%;" /></center>
##### Vanishing Gradient Problem

上面这个问题的原因不是overfitting，而是Vanishing Gradient(梯度消失)，解释如下：

当你把network叠得很深的时候，在靠近input的地方，这些参数的gradient(即对最后loss function的微分)是比较小的；而在比较靠近output的地方，它对loss的微分值会是比较大的

因此当你设定同样learning rate的时候，靠近input的地方，它参数的update是很慢的；而靠近output的地方，它参数的update是比较快的

所以在靠近input的地方，参数几乎还是random的时候，output就已经根据这些random的结果找到了一个local minima，然后就converge(收敛)了

这个时候你会发现，参数的loss下降的速度变得很慢，你就会觉得gradient已经接近于0了，于是把程序停掉了，由于这个converge，是几乎base on random的参数，所以model的参数并没有被训练充分，那在training data上得到的结果肯定是很差的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vanishing.png" width="60%;" /></center>
为什么会有这个现象发生呢？如果你自己把Backpropagation的式子写出来的话，就可以很轻易地发现用sigmoid function会导致这件事情的发生；但是，我们今天不看Backpropagation的式子，其实从直觉上来想你也可以了解这件事情发生的原因

某一个参数$w$对total cost $l$的偏微分，即gradient $\frac{\partial l}{\partial w}$，它直觉的意思是说，当我今天把这个参数做小小的变化的时候，它对这个cost的影响有多大；那我们就把第一个layer里的某一个参数$w$加上$\Delta w$，看看对network的output和target之间的loss有什么样的影响

$\Delta w$通过sigmoid function之后，得到output是会变小的，改变某一个参数的weight，会对某个neuron的output值产生影响，但是这个影响是会随着层数的递增而衰减的，sigmoid function的形状如下所示，它会把负无穷大到正无穷大之间的值都硬压到0~1之间，把较大的input压缩成较小的output

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/sigmoid-less.png" width="35%;" /></center>
因此即使$\Delta w$值很大，但每经过一个sigmoid function就会被缩小一次，所以network越深，$\Delta w$被衰减的次数就越多，直到最后，它对output的影响就是比较小的，相应的也导致input对loss的影响会比较小，于是靠近input的那些weight对loss的gradient $\frac{\partial l}{\partial w}$远小于靠近output的gradient

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vanish.png" width="60%;" /></center>
那怎么解决这个问题呢？比较早年的做法是去train RBM，它的精神就是，先把第一个layer train好，再去train第二个，然后再第三个...所以最后你在做Backpropagation的时候，尽管第一个layer几乎没有被train到，但一开始在做pre-train的时候就已经把它给train好了，这样RBM就可以在一定程度上解决问题

但其实改一下activation function可能就可以handle这个问题了

##### ReLU

###### introduction

现在比较常用的activation function叫做Rectified Linear Unit(整流线性单元函数，又称修正线性单元)，它的缩写是ReLU，该函数形状如下图所示，z为input，a为output，如果input>0则output = input，如果input<0则output = 0

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/ReLU1.png" width="60%;" /></center>
选择ReLU的理由如下：

- 跟sigmoid function比起来，ReLU的运算快很多
- ReLU的想法结合了生物上的观察( Pengel的paper )
- 无穷多bias不同的sigmoid function叠加的结果会变成ReLU
- ReLU可以处理Vanishing gradient的问题( the most important thing )

###### handle Vanishing gradient problem

下图是ReLU的neural network，以ReLU作为activation function的neuron，它的output要么等于0，要么等于input

当output=input的时候，这个activation function就是linear的；而output=0的neuron对整个network是没有任何作用的，因此可以把它们从network中拿掉

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/relu1.png" width="60%;" /></center>
拿掉所有output为0的neuron后如下图所示，此时整个network就变成了一个瘦长的**linear** network，linear的好处是，output=input，不会像sigmoid function一样使input产生的影响逐层递减

Q：这里就会有一个问题，我们之所以使用deep learning，就是因为想要一个non-linear、比较复杂的function，而使用ReLU不就会让它变成一个linear function吗？这样得到的function不是会变得很弱吗？

A：其实，使用ReLU之后的network整体来说还是non-linear的，如果你对input做小小的改变，不改变neuron的operation region的话，那network就是一个linear function；但是，如果你对input做比较大的改变，导致neuron的operation region被改变的话，比如从output=0转变到了output=input，network整体上就变成了non-linear function

注：这里的region是指input z<0和input z>0的两个范围

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/relu2.png" width="60%;" /></center>
Q：还有另外一个问题，我们对loss function做gradient descent，要求neural network是可以做微分的，但ReLU是一个分段函数，它是不能微分的(至少在z=0这个点是不可微的)，那该怎么办呢？

A：在实际操作上，当region的范围处于z>0时，微分值gradient就是1；当region的范围处于z<0时，微分值gradient就是0；当z为0时，就不要管它，相当于把它从network里面拿掉

###### ReLU-variant

其实ReLU还存在一定的问题，比如当input<0的时候，output=0，此时微分值gradient也为0，你就没有办法去update参数了，所以我们应该让input<0的时候，微分后还能有一点点的值，比如令$a=0.01z$，这个东西就叫做**Leaky ReLU**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/relu-variant.png" width="60%;" /></center>
既然a可以等于0.01z，那这个z的系数可不可以是0.07、0.08之类呢？所以就有人提出了**Parametric ReLU**，也就是令$a=\alpha \cdot z$，其中$\alpha$并不是固定的值，而是network的一个参数，它可以通过training data学出来，甚至每个neuron都可以有不同的$\alpha$值

这个时候又有人想，为什么一定要是ReLU这样子呢，activation function可不可以有别的样子呢？所以后来有了一个更进阶的想法，叫做**Maxout network**

##### Maxout

###### introduction

Maxout的想法是，让network自动去学习它的activation function，那Maxout network就可以自动学出ReLU，也可以学出其他的activation function，这一切都是由training data来决定的

假设现在有input $x_1,x_2$，它们乘上几组不同的weight分别得到5,7,-1,1，这些值本来是不同neuron的input，它们要通过activation function变为neuron的output；但在Maxout network里，我们事先决定好将某几个“neuron”的input分为一个group，比如5,7分为一个group，然后在这个group里选取一个最大值7作为output

这个过程就好像在一个layer上做Max Pooling一样，它和原来的network不同之处在于，它把原来几个“neuron”的input按一定规则组成了一个group，然后并没有使它们通过activation function，而是选取其中的最大值当做这几个“neuron”的output

当然，实际上原来的”neuron“早就已经不存在了，这几个被合并的“neuron”应当被看做是一个新的neuron，这个新的neuron的input是原来几个“neuron”的input组成的vector，output则取input的最大值，而并非由activation function产生

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout1.png" width="60%;" /></center>
在实际操作上，几个element被分为一个group这件事情是由你自己决定的，它就是network structure里一个需要被调的参数，不一定要跟上图一样两个分为一组

###### Maxout -> RELU

Maxout是如何模仿出ReLU这个activation function的呢？

下图左上角是一个ReLU的neuron，它的input x会乘上neuron的weight w，再加上bias b，然后通过activation function-ReLU，得到output a

- neuron的input为$z=wx+b$，为下图左下角紫线
- neuron的output为$a=z\ (z>0);\ a=0\ (z<0)$，为下图左下角绿线

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout2.png" width="60%;" /></center>
如果我们使用的是上图右上角所示的Maxout network，假设$z_1$的参数w和b与ReLU的参数一致，而$z_2$的参数w和b全部设为0，然后做Max Pooling，选取$z_1,z_2$较大值作为a

- neuron的input为$\begin{bmatrix}z_1 \ z_2 \end{bmatrix}$
    - $z_1=wx+b$，为上图右下角紫线
    - $z_2=0$，为上图右下角红线
- neuron的output为$\max{\begin{bmatrix}z_1 \ z_2 \end{bmatrix}}$，为上图右下角绿线

你会发现，此时ReLU和Maxout所得到的output是一模一样的，它们是相同的activation function

###### Maxout -> More than ReLU

除了ReLU，Maxout还可以实现更多不同的activation function

比如$z_2$的参数w和b不是0，而是$w',b'$，此时

- neuron的input为$\begin{bmatrix}z_1 \ z_2 \end{bmatrix}$
    - $z_1=wx+b$，为下图右下角紫线
    - $z_2=w'x+b'$，为下图右下角红线
- neuron的output为$\max{\begin{bmatrix}z_1 \ z_2 \end{bmatrix}}$，为下图右下角绿线

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout3.png" width="60%;" /></center>
这个时候你得到的activation function的形状(绿线形状)，是由network的参数$w,b,w',b'$决定的，因此它是一个**Learnable Activation Function**，具体的形状可以根据training data去generate出来

###### property

Maxout可以实现任何piecewise linear convex activation function(分段线性凸激活函数)，其中这个activation function被分为多少段，取决于你把多少个element z放到一个group里，下图分别是2个element一组和3个element一组的activation function的不同形状

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout4.png" width="60%;" /></center>
###### How to train Maxout

接下来我们要面对的是，怎么去train一个Maxout network，如何解决Max不能微分的问题

假设在下面的Maxout network中，红框圈起来的部分为每个neuron的output

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout-train.png" width="60%;" /></center>
其实Max operation就是linear的operation，只是它仅接在前面这个group里的某一个element上，因此我们可以把那些并没有被Max连接到的element通通拿掉，从而得到一个比较细长的linear network

实际上我们真正训练的并不是一个含有max函数的network，而是一个化简后如下图所示的linear network；当我们还没有真正开始训练模型的时候，此时这个network含有max函数无法微分，但是只要真的丢进去了一笔data，network就会马上根据这笔data确定具体的形状，此时max函数的问题已经被实际数据给解决了，所以我们完全可以根据这笔training data使用Backpropagation的方法去训练被network留下来的参数

所以我们担心的max函数无法微分，它只是理论上的问题；**在具体的实践上，我们完全可以先根据data把max函数转化为某个具体的函数，再对这个转化后的thiner linear network进行微分**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout-train2.png" width="60%;" /></center>
这个时候你也许会有一个问题，如果按照上面的做法，那岂不是只会train留在network里面的那些参数，剩下的参数该怎么办？那些被拿掉的直线(weight)岂不是永远也train不到了吗？

其实这也只是个理论上的问题，在实际操作上，我们之前已经提到过，每个linear network的structure都是由input的那一笔data来决定的，当你input不同data的时候，得到的network structure是不同的，留在network里面的参数也是不同的，**由于我们有很多很多笔training data，所以network的structure在训练中不断地变换，实际上最后每一个weight参数都会被train到**

所以，我们回到Max Pooling的问题上来，由于Max Pooling跟Maxout是一模一样的operation，既然如何训练Maxout的问题可以被解决，那训练Max Pooling又有什么困难呢？

**Max Pooling有关max函数的微分问题采用跟Maxout一样的方案即可解决**，至此我们已经解决了CNN部分的第一个问题

#### Adaptive learning rate

这个部分主要讲述的是关于Recipe of Deep Learning中Adaptive learning rate的一些理论

##### Review - Adagrad

我们之前已经了解过Adagrad的做法，让每一个parameter都要有不同的learning rate

Adagrad的精神是，假设我们考虑两个参数$w_1,w_2$，如果在$w_1$这个方向上，平常的gradient都比较小，那它是比较平坦的，于是就给它比较大的learning rate；反过来说，在$w_2$这个方向上，平常gradient都比较大，那它是比较陡峭的，于是给它比较小的learning rate

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/review-adagrad.png" width="60%;" /></center>
但我们实际面对的问题，很有可能远比Adagrad所能解决的问题要来的复杂，我们之前做Linear Regression的时候，我们做optimization的对象，也就是loss function，它是convex的形状；但实际上我们在做deep learning的时候，这个loss function可以是任何形状

##### RMSProp

###### learning rate

loss function可以是任何形状，对convex loss function来说，在每个方向上它会一直保持平坦或陡峭的状态，所以你只需要针对平坦的情况设置较大的learning rate，对陡峭的情况设置较小的learning rate即可

但是在下图所示的情况中，即使是在同一个方向上(如w1方向)，loss function也有可能一会儿平坦一会儿陡峭，所以你要随时根据gradient的大小来快速地调整learning rate

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rmsprop1.png" width="60%;" /></center>
所以真正要处理deep learning的问题，用Adagrad可能是不够的，你需要更dynamic的调整learning rate的方法，所以产生了Adagrad的进阶版——**RMSProp**

RMSprop还是一个蛮神奇的方法，因为它并不是在paper里提出来的，而是Hinton在mooc的course里面提出来的一个方法，所以需要cite(引用)的时候，要去cite Hinton的课程链接

###### how to do RMSProp

RMSProp的做法如下：

我们的learning rate依旧设置为一个固定的值 $\eta$ 除掉一个变化的值 $\sigma$，这个$\sigma$等于上一个$\sigma$和当前梯度$g$的加权方均根（特别的是，在第一个时间点，$\sigma^0$就是第一个算出来的gradient值$g^0$），即：
$$
w^{t+1}=w^t-\frac{\eta}{\sigma^t}g^t \\
\sigma^t=\sqrt{\alpha(\sigma^{t-1})^2+(1-\alpha)(g^t)^2}
$$
这里的$\alpha$值是可以自由调整的，RMSProp跟Adagrad不同之处在于，Adagrad的分母是对过程中所有的gradient取平方和开根号，也就是说Adagrad考虑的是整个过程平均的gradient信息；而RMSProp虽然也是对所有的gradient进行平方和开根号，但是它**用一个$\alpha$来调整对不同gradient的使用程度**，比如你把α的值设的小一点，意思就是你更倾向于相信新的gradient所告诉你的error surface的平滑或陡峭程度，而比较无视于旧的gradient所提供给你的information

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rmsprop2.png" width="60%;" /></center>
所以当你做RMSProp的时候，一样是在算gradient的root mean square，但是你可以给现在已经看到的gradient比较大的weight，给过去看到的gradient比较小的weight，来调整对gradient信息的使用程度

##### Momentum

###### optimization - local minima？

除了learning rate的问题以外，在做deep learning的时候，也会出现卡在local minimum、saddle point或是plateau的地方，很多人都会担心，deep learning这么复杂的model，可能非常容易就会被卡住了

但其实Yann LeCun在07年的时候，就提出了一个蛮特别的说法，他说你不要太担心local minima的问题，因为一旦出现local minima，它就必须在每一个dimension都是下图中这种山谷的低谷形状，假设山谷的低谷出现的概率为p，由于我们的network有非常非常多的参数，这里假设有1000个参数，每一个参数都要位于山谷的低谷之处，这件事发生的概率为$p^{1000}$，当你的network越复杂，参数越多，这件事发生的概率就越低

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/optimal1.png" width="60%;" /></center>
所以在一个很大的neural network里面，其实并没有那么多的local minima，搞不好它看起来其实是很平滑的，所以当你走到一个你觉得是local minima的地方被卡住了，那它八成就是global minima，或者是很接近global minima的地方

###### where is Momentum from

有一个heuristic(启发性)的方法可以稍微处理一下上面所说的“卡住”的问题，它的灵感来自于真实世界

假设在有一个球从左上角滚下来，它会滚到plateau的地方、local minima的地方，但是由于惯性它还会继续往前走一段路程，假设前面的坡没有很陡，这个球就很有可能翻过山坡，走到比local minima还要好的地方

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/physical.png" width="60%;" /></center>
所以我们要做的，就是把**惯性**塞到gradient descent里面，这件事情就叫做**Momentum**

###### how to do Momentum

当我们在gradient descent里加上Momentum的时候，每一次update的方向，不再只考虑gradient的方向，还要考虑上一次update的方向，那这里我们就用一个变量$v$去记录前一个时间点update的方向

随机选一个初始值$\theta^0$，初始化$v^0=0$，接下来计算$\theta^0$处的gradient，然后我们要移动的方向是由前一个时间点的移动方向$v^0$和gradient的反方向$\nabla L(\theta^0)$来决定的，即
$$
v^1=\lambda v^0-\eta \nabla L(\theta^0)
$$
注：这里的$\lambda$也是一个手动调整的参数，它表示惯性对前进方向的影响有多大

接下来我们第二个时间点要走的方向$v^2$，它是由第一个时间点移动的方向$v^1$和gradient的反方向$\nabla L(\theta^1)$共同决定的；$\lambda v$是图中的绿色虚线，它代表由于上一次的惯性想要继续走的方向；$\eta \nabla L(\theta)$是图中的红色虚线，它代表这次gradient告诉你所要移动的方向；它们的矢量和就是这一次真实移动的方向，为蓝色实线

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/momentum1.png" width="60%;" /></center>
gradient告诉我们走红色虚线的方向，惯性告诉我们走绿色虚线的方向，合起来就是走蓝色的方向

我们还可以用另一种方法来理解Momentum这件事，其实你在每一个时间点移动的步伐$v^i$，包括大小和方向，就是过去所有gradient的加权和

具体推导如下图所示，第一个时间点移动的步伐$v^1$是$\theta^0$处的gradient加权，第二个时间点移动的步伐$v^2$是$\theta^0$和$\theta^1$处的gradient加权和...以此类推；由于$\lambda$的值小于1，因此该加权意味着越是之前的gradient，它的权重就越小，也就是说，你更在意的是现在的gradient，但是过去的所有gradient也要对你现在update的方向有一定程度的影响力，这就是Momentum

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/momentum2.png" width="60%;" /></center>
如果你对数学公式不太喜欢的话，那我们就从直觉上来看一下加入Momentum之后是怎么运作的

在加入Momentum以后，每一次移动的方向，就是negative的gradient加上Momentum建议我们要走的方向，Momentum其实就是上一个时间点的movement

下图中，红色实线是gradient建议我们走的方向，直观上看就是根据坡度要走的方向；绿色虚线是Momentum建议我们走的方向，实际上就是上一次移动的方向；蓝色实线则是最终真正走的方向

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/momentum3.png" width="60%;" /></center>
如果我们今天走到local minimum的地方，此时gradient是0，红色箭头没有指向，它就会告诉你就停在这里吧，但是Momentum也就是绿色箭头，它指向右侧就是告诉你之前是要走向右边的，所以你仍然应该要继续往右走，所以最后你参数update的方向仍然会继续向右；你甚至可以期待Momentum比较强，惯性的力量可以支撑着你走出这个谷底，去到loss更低的地方

##### Adam

其实**RMSProp加上Momentum，就可以得到Adam**

根据下面的paper来快速描述一下Adam的algorithm：

- 先初始化$m_0=0$，$m_0$就是Momentum中，前一个时间点的movement

    再初始化$v_0=0$，$v_0$就是RMSProp里计算gradient的root mean square的$\sigma$

    最后初始化$t=0$，t用来表示时间点

- 先算出gradient $g_t$
    $$
    g_t=\nabla _{\theta}f_t(\theta_{t-1})
    $$

- 再根据过去要走的方向$m_{t-1}$和gradient $g_t$，算出现在要走的方向 $m_t$——Momentum
    $$
    m_t=\beta_1 m_{t-1}+(1-\beta_1) g_t
    $$

- 然后根据前一个时间点的$v_{t-1}$和gradient $g_t$的平方，算一下放在分母的$v_t$——RMSProp
    $$
    v_t=\beta_2 v_{t-1}+(1-\beta_2) g_t^2
    $$

- 接下来做了一个原来RMSProp和Momentum里没有的东西，就是bias correction，它使$m_t$和$v_t$都除上一个值，这个值本来比较小，后来会越来越接近于1 (原理详见paper)
    $$
    \hat{m}_t=\frac{m_t}{1-\beta_1^t} \\ \hat{v}_t=\frac{v_t}{1-\beta_2^t}
    $$

- 最后做update，把Momentum建议你的方向$\hat{m_t}$乘上learning rate $\alpha$，再除掉RMSProp normalize后建议的learning rate分母，然后得到update的方向
    $$
    \theta_t=\theta_{t-1}-\frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
    $$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adam.png" width="80%;" /></center>
### Good Results on Testing Data？

这一部分主要讲述如何在Testing data上得到更好的performance，分为三个模块，Early Stopping、Regularization和Dropout

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/result-test.png" width="60%;" /></center>
值得注意的是，Early Stopping和Regularization是很typical的做法，它们不是特别为deep learning所设计的；而Dropout是一个蛮有deep learning特色的做法

#### Early Stopping

假设你今天的learning rate调的比较好，那随着训练的进行，total loss通常会越来越小，但是Training set和Testing set的情况并不是完全一样的，很有可能当你在Training set上的loss逐渐减小的时候，在Testing set上的loss反而上升了

所以，理想上假如你知道testing data上的loss变化情况，你会在testing set的loss最小的时候停下来，而不是在training set的loss最小的时候停下来；但testing set实际上是未知的东西，所以我们需要用validation set来替代它去做这件事情

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/early-stop.png" width="60%;" /></center>
注：很多时候，我们所讲的“testing set”并不是指代那个未知的数据集，而是一些已知的被你拿来做测试之用的数据集，比如kaggle上的public set，或者是你自己切出来的validation set

#### Regularization

regularization就是在原来的loss function上额外增加几个term，比如我们要minimize的loss function原先应该是square error或cross entropy，那在做Regularization的时候，就在后面加一个Regularization的term

##### L2 regularization

regularization term可以是参数的L2 norm(L2正规化)，所谓的L2 norm，就是把model参数集$\theta$里的每一个参数都取平方然后求和，这件事被称作L2 regularization，即
$$
L2 \ regularization:||\theta||_2=(w_1)^2+(w_2)^2+...
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization1.png" width="60%;" /></center>
通常我们在做regularization的时候，新加的term里是不会考虑bias这一项的，因为加regularization的目的是为了让我们的function更平滑，而bias通常是跟function的平滑程度没有关系的

你会发现我们新加的regularization term $\lambda \frac{1}{2}||\theta||_2$里有一个$\frac{1}{2}$，由于我们是要对loss function求微分的，而新加的regularization term是参数$w_i$的平方和，对平方求微分会多出来一个系数2，我们的$\frac{1}{2}$就是用来和这个2相消的

L2 regularization具体工作流程如下：

- 我们加上regularization term之后得到了一个新的loss function：$L'(\theta)=L(\theta)+\lambda \frac{1}{2}||\theta||_2$
- 将这个loss function对参数$w_i$求微分：$\frac{\partial L'}{\partial w_i}=\frac{\partial L}{\partial w_i}+\lambda w_i$
- 然后update参数$w_i$：$w_i^{t+1}=w_i^t-\eta \frac{\partial L'}{\partial w_i}=w_i^t-\eta(\frac{\partial L}{\partial w_i}+\lambda w_i^t)=(1-\eta \lambda)w_i^t-\eta \frac{\partial L}{\partial w_i}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization2.png" width="60%;" /></center>
如果把这个推导出来的式子和原式作比较，你会发现参数$w_i$在每次update之前，都会乘上一个$(1-\eta \lambda)$，而$\eta$和$\lambda$通常会被设为一个很小的值，因此$(1-\eta \lambda)$通常是一个接近于1的值，比如0.99,；也就是说，regularization做的事情是，每次update参数$w_i$之前，不分青红皂白就先对原来的$w_i$乘个0.99，这意味着，随着update次数增加，参数$w_i$会越来越接近于0

Q：你可能会问，要是所有的参数都越来越靠近0，那最后岂不是$w_i$通通变成0，得到的network还有什么用？

A：其实不会出现最后所有参数都变为0的情况，因为通过微分得到的$\eta \frac{\partial L}{\partial w_i}$这一项是会和前面$(1-\eta \lambda)w_i^t$这一项最后取得平衡的

使用L2 regularization可以让weight每次都变得更小一点，这就叫做**Weight Decay**(权重衰减)

##### L1 regularization

除了L2 regularization中使用平方项作为new term之外，还可以使用L1 regularization，把平方项换成每一个参数的绝对值，即
$$
||\theta||_1=|w_1|+|w_2|+...
$$
Q：你的第一个问题可能会是，绝对值不能微分啊，该怎么处理呢？

A：实际上绝对值就是一个V字形的函数，在V的左边微分值是-1，在V的右边微分值是1，只有在0的地方是不能微分的，那真的走到0的时候就胡乱给它一个值，比如0，就ok了

如果w是正的，那微分出来就是+1，如果w是负的，那微分出来就是-1，所以这边写了一个w的sign function，它的意思是说，如果w是正数的话，这个function output就是+1，w是负数的话，这个function output就是-1

L1 regularization的工作流程如下：

- 我们加上regularization term之后得到了一个新的loss function：$L'(\theta)=L(\theta)+\lambda \frac{1}{2}||\theta||_1$
- 将这个loss function对参数$w_i$求微分：$\frac{\partial L'}{\partial w_i}=\frac{\partial L}{\partial w_i}+\lambda \ sgn(w_i)$
- 然后update参数$w_i$：$w_i^{t+1}=w_i^t-\eta \frac{\partial L'}{\partial w_i}=w_i^t-\eta(\frac{\partial L}{\partial w_i}+\lambda \ sgn(w_i^t))=w_i^t-\eta \frac{\partial L}{\partial w_i}-\eta \lambda \ sgn(w_i^t)$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization3.png" width="60%;" /></center>
这个式子告诉我们，每次update的时候，不管三七二十一都要减去一个$\eta \lambda \ sgn(w_i^t)$，如果w是正的，sgn是+1，就会变成减一个positive的值让你的参数变小；如果w是负的，sgn是-1，就会变成加一个值让你的参数变大；总之就是让它们的绝对值减小至接近于0

##### L1 V.s. L2

我们来对比一下L1和L2的update过程：
$$
L1: w_i^{t+1}=w_i^t-\eta \frac{\partial L}{\partial w_i}-\eta \lambda \ sgn(w_i^t)\\
L2: w_i^{t+1}=(1-\eta \lambda)w_i^t-\eta \frac{\partial L}{\partial w_i}
$$
L1和L2，虽然它们同样是让参数的绝对值变小，但它们做的事情其实略有不同：

- L1使参数绝对值变小的方式是每次update**减掉一个固定的值**
- L2使参数绝对值变小的方式是每次update**乘上一个小于1的固定值**

因此，当参数w的绝对值比较大的时候，L2会让w下降得更快，而L1每次update只让w减去一个固定的值，train完以后可能还会有很多比较大的参数；当参数w的绝对值比较小的时候，L2的下降速度就会变得很慢，train出来的参数平均都是比较小的，而L1每次下降一个固定的value，train出来的参数是比较sparse的，这些参数有很多是接近0的值，也会有很大的值

在之前所讲的CNN的task里，用L1做出来的效果是比较合适的，是比较sparse的

##### Weight Decay

之前提到了Weight Decay，那实际上我们在人脑里面也会做Weight Decay

下图分别描述了，刚出生的时候，婴儿的神经是比较稀疏的；6岁的时候，就会有很多很多的神经；但是到14岁的时候，神经间的连接又减少了，所以neural network也会跟我们人有一些很类似的事情，如果有一些weight你都没有去update它，那它每次都会越来越小，最后就接近0然后不见了

这跟人脑的运作，是有异曲同工之妙

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization4.png" width="60%;" /></center>
#### some tips

ps：在deep learning里面，regularization虽然有帮助，但它的重要性往往没有SVM这类方法来得高，因为我们在做neural network的时候，通常都是从一个很小的、接近于0的值开始初始参数的，而做update的时候，通常都是让参数离0越来越远，但是regularization要达到的目的，就是希望我们的参数不要离0太远

如果你做的是Early Stopping，它会减少update的次数，其实也会避免你的参数离0太远，这跟regularization做的事情是很接近的

所以在neural network里面，regularization的作用并没有SVM来的重要，SVM其实是explicitly把regularization这件事情写在了它的objective function(目标函数)里面，SVM是要去解一个convex optimization problem，因此它解的时候不一定会有iteration的过程，它不会有Early Stopping这件事，而是一步就可以走到那个最好的结果了，所以你没有办法用Early Stopping防止它离目标太远，你必须要把regularization explicitly加到你的loss function里面去

#### Dropout

这里先讲dropout是怎么做的，然后再来解释为什么这样做

##### How to do Dropout

Dropout是怎么做的呢？

###### Training

在training的时候，每次update参数之前，我们对每一个neuron(也包括input layer的“neuron”)做sampling(抽样) ，每个neuron都有p%的几率会被丢掉，如果某个neuron被丢掉的话，跟它相连的weight也都要被丢掉

实际上就是每次update参数之前都通过抽样只保留network中的一部分neuron来做训练

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout1.png" width="60%;" /></center>
做完sampling以后，network structure就会变得比较细长了，然后你再去train这个细长的network

注：每次update参数之前都要做一遍sampling，所以每次update参数的时候，拿来training的network structure都是不一样的；你可能会觉得这个方法跟前面提到的Maxout会有一点像，但实际上，Maxout是每一笔data对应的network structure不同，而Dropout是每一次update的network structure都是不同的(每一个minibatch对应着一次update，而一个minibatch里含有很多笔data)

当你在training的时候使用dropout，得到的performance其实是会变差的，因为某些neuron在training的时候莫名其妙就会消失不见，但这并不是问题，因为：

==**Dropout真正要做的事情，就是要让你在training set上的结果变差，但是在testing set上的结果是变好的**==

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout2.png" width="60%;" /></center>
所以如果你今天遇到的问题是在training set上得到的performance不够好，你再加dropout，就只会越做越差；这告诉我们，不同的problem需要用不同的方法去解决，而不是胡乱使用，dropout就是针对testing set的方法，当然不能够拿来解决training set上的问题啦！

###### Testing

在使用dropout方法做testing的时候要注意两件事情：

- testing的时候不做dropout，所有的neuron都要被用到
- 假设在training的时候，dropout rate是p%，从training data中被learn出来的所有weight都要乘上(1-p%)才能被当做testing的weight使用

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout3.png" width="60%;" /></center>
##### Why Dropout？

###### 为什么dropout会有用？

直接的想法是这样子：

在training的时候，会丢掉一些neuron，就好像是你要练轻功的时候，会在脚上绑一些重物；然后，你在实际战斗的时候，就是实际testing的时候，是没有dropout的，就相当于把重物拿下来，所以你就会变得很强

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout4.png" width="60%;" /></center>
另一个直觉的理由是这样，neural network里面的每一个neuron就是一个学生，那大家被连接在一起就是大家听到说要组队做final project，那在一个团队里总是有人会拖后腿，就是他会dropout，所以假设你觉得自己的队友会dropout，这个时候你就会想要好好做，然后去carry这个队友，这就是training的过程

那实际在testing的时候，其实大家都有好好做，没有人需要被carry，由于每个人都比一般情况下更努力，所以得到的结果会是更好的，这也就是testing的时候不做dropout的原因

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout5.png" width="60%;" /></center>
###### 为什么training和testing使用的weight是不一样的呢？

直觉的解释是这样的：

假设现在的dropout rate是50%，那在training的时候，你总是期望每次update之前会丢掉一半的neuron，就像下图左侧所示，在这种情况下你learn好了一组weight参数，然后拿去testing

但是在testing的时候是没有dropout的，所以如果testing使用的是和training同一组weight，那左侧得到的output z和右侧得到的output z‘，它们的值其实是会相差两倍的，即$z'≈2z$，这样会造成testing的结果与training的结果并不match，最终的performance反而会变差

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout6.png" width="60%;" /></center>
那这个时候，你就需要把右侧testing中所有的weight乘上0.5，然后做normalization，这样z就会等于z'，使得testing的结果与training的结果是比较match的

##### Dropout is a kind of ensemble

在文献上有很多不同的观点来解释为什么dropout会work，其中一种比较令人信服的解释是：**dropout是一种终极的ensemble的方法**

###### ensemble精神的解释

ensemble的方法在比赛的时候经常用得到，它的意思是说，我们有一个很大的training set，那你每次都只从这个training set里面sample一部分的data出来，像下图一样，抽取了set1,set2,set3,set4

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout7.png" width="60%;" /></center>
我们之前在讲bias和variance的trade off的时候说过，打靶有两种情况：

- 一种是因为bias大而导致打不准(参数过少)
- 另一种是因为variance大而导致打不准(参数过多)

假设我们今天有一个很复杂的model，它往往是bias比较准，但variance很大的情况，如果你有很多个笨重复杂的model，虽然它们的variance都很大，但最后平均起来，结果往往就会很准

所以ensemble做的事情，就是利用这个特性，我们从原来的training data里面sample出很多subset，然后train很多个model，每一个model的structure甚至都可以不一样；在testing的时候，丢了一笔testing data进来，使它通过所有的model，得到一大堆的结果，然后把这些结果平均起来当做最后的output

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout8.png" width="60%;" /></center>
如果你的model很复杂，这一招往往是很有用的，那著名的random forest(随机森林)也是实践这个精神的一个方法，也就是如果你用一个decision tree，它就会很弱，也很容易overfitting，而如果采用random forest，它就没有那么容易overfitting

###### 为什么dropout是一个终极的ensemble方法呢？

在training network的时候，每次拿一个minibatch出来就做一次update，而根据dropout的特性，每次update之前都要对所有的neuron进行sample，因此每一个minibatch所训练的network都是不同的

假设我们有M个neuron，每个neuron都有可能drop或不drop，所以总共可能的network数量有$2^M$个；所以当你在做dropout的时候，相当于是在用很多个minibatch分别去训练很多个network(一个minibatch一般设置为100笔data)，由于update次数是有限的，所以做了几次update，就相当于train了几个不同的network，最多可以训练到$2^M$个network

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout9.png" width="60%;" /></center>
每个network都只用一个minibatch的data来train，可能会让人感到不安，一个batch才100笔data，怎么train一个network呢？其实没有关系，因为这些**不同的network之间的参数是shared**，也就是说，虽然一个network只能用一个minibatch来train，但同一个weight可以在不同的network里被不同的minibatch train，所以同一个weight实际上是被所有没有丢掉它的network一起share的，它是拿所有这些network的minibatch合起来一起train的结果

###### 实际操作ensemble的做法

那按照ensemble这个方法的逻辑，在testing的时候，你把那train好的一大把network通通拿出来，然后把手上这一笔testing data丢到这把network里面去，每个network都给你吐出一个结果来，然后你把所有的结果平均起来 ，就是最后的output

但是在实际操作上，如下图左侧所示，这一把network实在太多了，你没有办法每一个network都丢一个input进去，再把它们的output平均起来，这样运算量太大了

所以dropout最神奇的地方是，当你并没有把这些network分开考虑，而是用一个完整的network，这个network的weight是用之前那一把network train出来的对应weight乘上(1-p%)，然后再把手上这笔testing data丢进这个完整的network，得到的output跟network分开考虑的ensemble的output，是惊人的相近

也就是说下图左侧ensemble的做法和右侧dropout的做法，得到的结果是approximate(近似)的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout10.png" width="60%;" /></center>
###### 举例说明dropout和ensemble的关系

这里用一个例子来解释：

我们train一个下图右上角所示的简单的network，它只有一个neuron，activation function是linear的，并且不考虑bias，这个network经过dropout训练以后得到的参数分别为$w_1,w_2$，那给它input $x_1,x_2$，得到的output就是$z=w_1 x_1+w_2 x_2$

如果我们今天要做ensemble的话，theoretically就是像下图这么做，每一个neuron都有可能被drop或不drop，这里只有两个input的neuron，所以我们一共可以得到2^2=4种network

我们手上这笔testing data $x_1,x_2$丢到这四个network中，分别得到4个output：$w_1x_1+w_2x_2,w_2x_2,w_1x_1,0$，然后根据ensemble的精神，把这四个network的output通通都average起来，得到的结果是$\frac{1}{2}(w_1x_1+w_2x_2)$

那根据dropout的想法，我们把从training中得到的参数$w_1,w_2$乘上(1-50%)，作为testing network里的参数，也就是$w'_1,w'_2=(1-50\%)(w_1,w_2)=0.5w_1,0.5w_2$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout11.png" width="60%;" /></center>
这边想要呈现的是，在这个最简单的case里面，用不同的network structure做ensemble这件事情，跟我们用一整个network，并且把weight乘上一个值而不做ensemble所得到的output，其实是一样的

值得注意的是，**只有是linear的network，才会得到上述的等价关系**，如果network是非linear的，ensemble和dropout是不equivalent的；但是，dropout最后一个很神奇的地方是，虽然在non-linear的情况下，它是跟ensemble不相等的，但最后的结果还是会work

==**如果network很接近linear的话，dropout所得到的performance会比较好，而ReLU和Maxout的network相对来说是比较接近于linear的，所以我们通常会把含有ReLU或Maxout的network与Dropout配合起来使用**==