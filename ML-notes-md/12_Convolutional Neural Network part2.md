# Convolutional Neural Network part2

> 人们常常会说，deep learning就是一个黑盒子，你learn完以后根本就不知道它得到了什么，所以会有很多人不喜欢这种方法，这篇文章就讲述了三个问题：What does CNN do？Why CNN？How to design CNN?

#### What does CNN learn？

##### what is intelligent

如果今天有一个方法，它可以让你轻易地理解为什么这个方法会下这样的判断和决策的话，那其实你会觉得它不够intelligent；它必须要是你无法理解的东西，这样它才够intelligent，至少你会感觉它很intelligent

所以，大家常说deep learning就是一个黑盒子，你learn出来以后，根本就不知道为什么是这样子，于是你会感觉它很intelligent，但是其实还是有很多方法可以分析的，今天我们就来示范一下怎么分析CNN，看一下它到底学到了什么

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-keras3.png" width="60%;"></center>
要分析第一个convolution的filter是比较容易的，因为第一个convolution layer里面，每一个filter就是一个3\*3的matrix，它对应到3\*3范围内的9个pixel，所以你只要看这个filter的值，就可以知道它在detect什么东西，因此第一层的filter是很容易理解的

但是你比较没有办法想像它在做什么事情的，是第二层的filter，它们是50个同样为3\*3的filter，但是这些filter的input并不是pixel，而是做完convolution再做Max pooling的结果，因此filter考虑的范围并不是3\*3=9个pixel，而是一个长宽为3\*3，高为25的cubic，filter实际在image上看到的范围是远大于9个pixel的，所以你就算把它的weight拿出来，也不知道它在做什么

##### what does filter do

那我们怎么来分析一个filter它做的事情是什么呢？你可以这样做：

我们知道在第二个convolution layer里面的50个filter，每一个filter的output就是一个11\*11的matrix，假设我们现在把第k个filter的output拿出来，如下图所示，这个matrix里的每一个element，我们叫它$a^k_{ij}$，上标k表示这是第k个filter，下标ij表示它在这个matrix里的第i个row，第j个column

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/kth-filter.png" width="60%;" /></center>
接下来我们define一个$a^k$叫做**Degree of the activation of the k-th filter**，这个值表示现在的第k个filter，它有多被activate，有多被“启动”，直观来讲就是描述现在input的东西跟第k个filter有多接近，它对filter的激活程度有多少

第k个filter被启动的degree $a^k$就定义成，它与input进行卷积所输出的output里所有element的summation，以上图为例，就是这11*11的output matrix里所有元素之和，用公式描述如下：
$$
a^k=\sum\limits^{11}_{i=1}\sum\limits^{11}_{j=1} a^k_{ij}
$$
也就是说，我们input一张image，然后把这个filter和image进行卷积所output的11\*11个值全部加起来，当作现在这个filter被activate的程度

接下来我们要做的事情是这样子，我们想要知道第k个filter的作用是什么，那我们就要找一张image，这张image可以让第k个filter被activate的程度最大；于是我们现在要解的问题是，找一个image x，它可以让我们定义的activation的degree $a^k$最大，即：
$$
x^*=\arg \max\limits_x a^k
$$
之前我们求minimize用的是gradient descent，那现在我们求Maximum用gradient ascent(梯度上升法)就可以做到这件事了

仔细一想这个方法还是颇为神妙的，因为我们现在是把input x作为要找的参数，对它去用gradient descent或ascent进行update，原来在train CNN的时候，input是固定的，model的参数是要用gradient descent去找出来的；但是现在这个立场是反过来的，在这个task里面model的参数是固定的，我们要用gradient ascent去update这个x，让它可以使degree of activation最大

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/gradient-ascent.png" width="60%;" /></center>
上图就是得到的结果，50个filter理论上可以分别找50张image使对应的activation最大，这里仅挑选了其中的12张image作为展示，这些image有一个共同的特征，它们里面都是一些**反复出现的某种texture(纹路)**，比如说第三张image上布满了小小的斜条纹，这意味着第三个filter的工作就是detect图上有没有斜条纹，要知道现在每个filter检测的都只是图上一个小小的范围而已，所以图中一旦出现一个小小的斜条纹，这个filter就会被activate，相应的output也会比较大，所以如果整张image上布满这种斜条纹的话，这个时候它会最兴奋，filter的activation程度是最大的，相应的output值也会达到最大

因此每个filter的工作就是去detect某一种pattern，detect某一种线条，上图所示的filter所detect的就是不同角度的线条，所以今天input有不同线条的话，某一个filter会去找到让它兴奋度最高的匹配对象，这个时候它的output就是最大的

##### what does neuron do

我们做完convolution和max pooling之后，会将结果用Flatten展开，然后丢到Fully connected的neural network里面去，之前已经搞清楚了filter是做什么的，那我们也想要知道在这个neural network里的每一个neuron是做什么的，所以就对刚才的做法如法炮制

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/neuron-do.png" width="60%;" /></center>
我们定义第j个neuron的output就是$a_j$，接下来就用gradient ascent的方法去找一张image x，把它丢到neural network里面就可以让$a_j$的值被maximize，即：
$$
x^*=\arg \max\limits_x a^j
$$
找到的结果如上图所示，同理这里仅取出其中的9张image作为展示，你会发现这9张图跟之前filter所观察到的情形是很不一样的，刚才我们观察到的是类似纹路的东西，那是因为每个filter考虑的只是图上一部分的vision，所以它detect的是一种texture；但是在做完Flatten以后，每一个neuron不再是只看整张图的一小部分，它现在的工作是看整张图，所以对每一个neuron来说，让它最兴奋的、activation最大的image，不再是texture，而是一个完整的图形

##### what about output

接下来我们考虑的是CNN的output，由于是手写数字识别的demo，因此这里的output就是10维，我们把某一维拿出来，然后同样去找一张image x，使这个维度的output值最大，即
$$
x^*=\arg \max_x y^i
$$
你可以想象说，既然现在每一个output的每一个dimension就对应到一个数字，那如果我们去找一张image x，它可以让对应到数字1的那个output layer的neuron的output值最大，那这张image显然应该看起来会像是数字1，你甚至可以期待，搞不好用这个方法就可以让machine自动画出数字

但实际上，我们得到的结果是这样子，如下图所示

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-output.png" width="60%;" /></center>
上面的每一张图分别对应着数字0-8，你会发现，可以让数字1对应neuron的output值最大的image其实长得一点也不像1，就像是电视机坏掉的样子，为了验证程序有没有bug，这里又做了一个实验，把上述得到的image真的作为testing data丢到CNN里面，结果classify的结果确实还是认为这些image就对应着数字0-8

所以今天这个neural network，它所学到的东西跟我们人类一般的想象认知是不一样的

那我们有没有办法，让上面这个图看起来更像数字呢？想法是这样的，我们知道一张图是不是一个数字，它会有一些基本的假设，比如这些image，你不知道它是什么数字，你也会认为它显然就不是一个digit，因为人类手写出来的东西就不是长这个样子的，所以我们要对这个x做一些regularization，我们要对找出来的x做一些constraint(限制约束)，我们应该告诉machine说，虽然有一些x可以让你的y很大，但是它们不是数字

那我们应该加上什么样的constraint呢？最简单的想法是说，画图的时候，白色代表的是有墨水、有笔画的地方，而对于一个digit来说，整张image上涂白的区域是有限的，像上面这些整张图都是白白的，它一定不会是数字

假设image里的每一个pixel都用$x_{ij}$表示，我们把所有pixel值取绝对值并求和，也就是$\sum\limits_{i,j}|x_{ij}|$，这一项其实就是之前提到过的L1的regularization，再用$y^i$减去这一项，得到
$$
x^*=\arg \max\limits_x (y^i-\sum\limits_{i,j} |x_{ij}|)
$$
这次我们希望再找一个input x，它可以让$y^i$最大的同时，也要让$|x_ij|$的summation越小越好，也就是说我们希望找出来的image，大部分的地方是没有涂颜色的，只有少数数字笔画在的地方才有颜色出现

加上这个constraint以后，得到的结果会像下图右侧所示一样，已经隐约有些可以看出来是数字的形状了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/L1.png" width="60%;" /></center>
如果再加上一些额外的constraint，比如你希望相邻的pixel是同样的颜色等等，你应该可以得到更好的结果

#### Deep Dream

其实，这就是Deep Dream的精神，Deep Dream是说，如果你给machine一张image，它会在这个image里面加上它看到的东西

怎么做这件事情呢？你就找一张image丢到CNN里面去，然后你把某一个convolution layer里面的filter或是fully connected layer里的某一个hidden layer的output拿出来，它其实是一个vector；接下来把本来是positive的dimension值调大，negative的dimension值调小，也就是让正的更正，负的更负，然后把它作为新的image的目标

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-dream.png" width="60%;" /></center>
这里就是把3.9、2.3的值调大，-1.5的值调小，总体来说就是使它们的绝对值变大，然后用gradient descent的方法找一张image x，让它通过这个hidden layer后的output就是你调整后的target，这么做的目的就是，**让CNN夸大化它看到的东西**——make CNN exaggerates what is sees

也就是说，如果某个filter有被activate，那你让它被activate的更剧烈，CNN可能本来看到了某一样东西，那现在你就让它看起来更像原来看到的东西，这就是所谓的**夸大化**

如果你把上面这张image拿去做Deep Dream的话，你看到的结果就会像下面这个样子

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-dream2.png" width="60%;" /></center>

就好像背后有很多念兽，要凝才看得到，比如像上图右侧那一只熊，它原来是一个石头，对机器来说，它看这张图的时候，本来就觉得这个石头有点像熊，所以你就更强化这件事，让它看起来真的就变成了一只熊，这个就是Deep Dream

#### Deep Style

Deep Dream还有一个进阶的版本，就叫做Deep Style，如果今天你input一张image，Deep Style做的事情就是让machine去修改这张图，让它有另外一张图的风格，如下所示

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-style.png" width="60%;" /></center>

实际上机器做出来的效果惊人的好，具体的做法参考reference：[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

这里仅讲述Deep Style的大致思路，你把原来的image丢给CNN，得到CNN filter的output，代表这样image里面有什么样的content，然后你把呐喊这张图也丢到CNN里面得到filter的output，注意，我们并不在于一个filter output的value到底是什么，一个单独的数字并不能代表任何的问题，我们真正在意的是，filter和filter的output之间的correlation，这个correlation代表了一张image的style

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-style2.png" width="60%;" /></center>
接下来你就再用一个CNN去找一张image，**这张image的content像左边的图片**，比如这张image的filter output的value像左边的图片；同时让**这张image的style像右边的图片**，所谓的style像右边的图片是说，这张image output的filter之间的correlation像右边这张图片

最终你用gradient descent找到一张image，同时可以maximize左边的content和右边的style，它的样子就像上图左下角所示

#### More Application——Playing Go

##### What does CNN do in Playing Go

CNN可以被运用到不同的应用上，不只是影像处理，比如出名的alphaGo

想要让machine来下围棋，不见得要用CNN，其实一般typical的neural network也可以帮我们做到这件事情

你只要learn一个network，也就是找一个function，它的input是棋盘当前局势，output是你下一步根据这个棋盘的盘势而应该落子的位置，这样其实就可以让machine学会下围棋了，所以用fully connected的feedforward network也可以做到让machine下围棋这件事情

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/play-go.png" width="60%;" /></center>
也就是说，你只要告诉它input是一个19\*19的vector，vector的每一个dimension对应到棋盘上的某一个位置，如果那一个位置有一个黑子的话，就是1，如果有一个白子的话，就是-1，反之呢，就是0，所以如果你把棋盘描述成一个19\*19的vector，丢到一个fully connected的feedforward network里，output也是19\*19个dimension ，每一个dimension对应到棋盘上的一个位置，那machine就可以学会下围棋了

但实际上如果我们采用CNN的话，会得到更好的performance，我们之前举的例子都是把CNN用在图像上面，也就是input是一个matrix，而棋盘其实可以很自然地表示成一个19\*19的matrix，那对CNN来说，就是直接把它当成一个image来看待，然后再output下一步要落子的位置，具体的training process是这样的：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/go-process.png" width="60%;" /></center>
你就搜集很多棋谱，比如说上图这个是进藤光和社青春的棋谱，初手下在5之五，次手下在天元，然后再下在5之五，接下来你就告诉machine说，看到落子在5之五，CNN的output就是天元的地方是1，其他的output是0；看到5之五和天元都有子，那你的output就是5之五的地方是1，其他都是0

上面是supervised的部分，那其实呢AlphaGo还有reinforcement learning的部分，这个后面的章节会讲到

##### Why CNN for Playing Go

自从AlphaGo用了CNN以后，大家都觉得好像CNN应该很厉害，所以有时候如果你没有用CNN来处理问题，人家就会来问你；比如你去面试的时候，你的硕士论文里面没有用CNN来处理问题，口试的人可能不知道CNN是什么 ，但是他就会问你说为什么不用CNN呢，CNN不是比较强吗？这个时候如果你真的明白了为什么要用CNN，什么时候才要用CNN这个问题，你就可以直接给他怼回去

那什么时候我们可以用CNN呢？你要有image该有的那些特性，也就是上一篇文章开头所说的，根据观察到的三个property，我们才设计出了CNN这样的network架构：

- **Some patterns are much smaller than the whole image**
- **The same patterns appear in different regions**
- **Subsampling the pixels will not change the object**

CNN能够应用在Alpha-Go上，是因为围棋有一些特性和图像处理是很相似的

在property 1，有一些pattern是比整张image要小得多，在围棋上，可能也有同样的现象，比如下图中一个白子被3个黑子围住，这个叫做吃，如果下一个黑子落在白子下面，就可以把白子提走；只有另一个白子接在下面，它才不会被提走

那现在你只需要看这个小小的范围，就可以侦测这个白子是不是属于被叫吃的状态，你不需要看整个棋盘，才知道这件事情，所以这件事情跟image有着同样的性质；在AlphaGo里面，它第一个layer其实就是用5\*5的filter，显然做这个设计的人，觉得围棋上最基本的pattern可能都是在5\*5的范围内就可以被侦测出来

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/why-cnn-go.png" width="60%;" /></center>
在property 2，同样的pattern可能会出现在不同的region，在围棋上也可能有这个现象，像这个叫吃的pattern，它可以出现在棋盘的左上角，也可以出现在右下角，它们都是叫吃，都代表了同样的意义，所以你可以用同一个detector，来处理这些在不同位置的同样的pattern

所以对围棋来说呢，它在第一个observation和第二个observation是有这个image的特性的，但是，让我们没有办法想通的地方，就是第三点

##### Max Pooling for Alpha Go？——read alpha-go paper

我们可以对一个image做subsampling，你拿掉奇数行、偶数列的pixel，把image变成原来的1/4的大小也不会影响你看这张图的样子，基于这个观察才有了Max pooling这个layer；但是，对围棋来说，它可以做这件事情吗？比如说，你对一个棋盘丢掉奇数行和偶数列，那它还和原来是同一个函式吗？显然不是的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/go-property3.png" width="60%;" /></center>
如何解释在棋盘上使用Max Pooling这件事情呢？有一些人觉得说，因为AlphaGo使用了CNN，它里面有可能用了Max pooling这样的构架，所以，或许这是它的一个弱点，你要是针对这个弱点攻击它，也许就可以击败它

AlphaGo的paper内容不多，只有6页左右，它只说使用了CNN，却没有在正文里面仔细地描述它的CNN构架，但是在这篇paper长长附录里，其实是有描述neural network structure的，如上图所示

它是这样说的，input是一个19\*19\*48的image，其中19\*19是棋盘的格局，对Alpha来说，每一个位置都用48个value来描述，这是因为加上了domain knowledge，它不只是描述某位置有没有白子或黑子，它还会观察这个位置是不是处于叫吃的状态等等

先用一个hidden layer对image做zero padding，也就是把原来19\*19的image外围补0，让它变成一张23\*23的image，然后使用k个5\*5的filter对该image做convolution，stride设为1，activation function用的是ReLU，得到的output是21\*21的image；接下来使用k个3\*3的filter，stride设为1，activation function还是使用ReLU，...

你会发现这个AlphaGo的network structure一直在用convolution，其实**根本就没有使用Max Pooling**，原因并不是疏失了什么之类的，而是根据围棋的特性，我们本来就不需要在围棋的CNN里面，用Max pooling这样的构架

举这个例子是为了告诉大家：

==**neural network架构的设计，是应用之道，存乎一心**==

#### More Application——Speech、Text

##### Speech 

CNN也可以用在很多其他的task里面，比如语音处理上，我们可以把一段声音表示成spectrogram，spectrogram的横轴是时间，纵轴则是这一段时间里声音的频率

下图中是一段“你好”的音频，偏红色代表这段时间里该频率的energy是比较大的，也就对应着“你”和“好”这两个字，也就是说spectrogram用颜色来描述某一个时刻不同频率的能量

我们也可以让机器把这个spectrogram就当作一张image，然后用CNN来判断说，input的这张image对应着什么样的声音信号，那通常用来判断结果的单位，比如phoneme，就是类似音标这样的单位

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-speech.png" width="60%;" /></center>
这边比较神奇的地方就是，当我们把一段spectrogram当作image丢到CNN里面的时候，在语音上，我们通常只考虑在frequency(频率)方向上移动的filter，我们的filter就像上图这样，是长方形的，它的宽就跟image的宽是一样的，并且**filter只在Frequency即纵坐标的方向上移动，而不在时间的序列上移动**

这是因为在语音里面，CNN的output后面都还会再接别的东西，比如接LSTM之类，它们都已经有考虑typical的information，所以你在CNN里面再考虑一次时间的information其实没有什么特别的帮助，但是为什么在频率上 的filter有帮助呢？

我们用CNN的目的是为了用同一个filter把相同的pattern给detect出来，在声音讯号上，虽然男生和女生说同样的话看起来这个spectrogram是非常不一样的，但实际上他们的不同只是表现在一个频率的shift而已(整体在频率上的位移)，男生说的“你好”跟女生说的“你好”，它们的pattern其实是一样的，比如pattern是spectrogram变化的情形，男生女生的声音的变化情况可能是一样的，它们的差别可能只是所在的频率范围不同而已，所以filter在frequency的direction上移动是有效的

所以，这又是另外一个例子，当你把CNN用在一个Application的时候呢，你永远要想一想这个Application的特性是什么，根据这个特性你再去design network的structure，才会真正在理解的基础上去解决问题

##### Text

CNN也可以用在文字处理上，假设你的input是一个word sequence，你要做的事情是让machine侦测这个word sequence代表的意思是positive的还是negative的

首先你把这个word sequence里面的每一个word都用一个vector来表示，vector代表的这个word本身的semantic (语义)，那如果两个word本身含义越接近的话，它们的vector在高维的空间上就越接近，这个东西就叫做word embedding

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-text.png" width="60%;" /></center>
把一个sentence里面所有word的vector排在一起，它就变成了一张image，你把CNN套用到这个image上，那filter的样子就是上图蓝色的matrix，它的高和image的高是一样的，然后把filter沿着句子里词汇的顺序来移动，每个filter移动完成之后都会得到一个由内积结果组成的vector，不同的filter就会得到不同的vector，接下来做Max pooling，然后把Max pooling的结果丢到fully connected layer里面，你就会得到最后的output

与语音处理不同的是，**在文字处理上，filter只在时间的序列(按照word的顺序)上移动，而不在这个embedding的dimension上移动**；因为在word embedding里面，不同dimension是independent的，它们是相互独立的，不会出现有两个相同的pattern的情况，所以在这个方向上面移动filter，是没有意义的

所以这又是另外一个例子，虽然大家觉得CNN很powerful，你可以用在各个不同的地方，但是当你应用到一个新的task的时候，你要想一想这个新的task在设计CNN的构架的时候，到底该怎么做



#### conclusion

本文的重点在于CNN的theory base，也就是What is CNN？What does CNN do？Why CNN？总结起来就是三个property、两个架构和一个理念，这也是使用CNN的条件基础：

##### 三个property

- **Some patterns are much smaller than the whole image** ——property 1
- **The same patterns appear in different regions** ——property 2
- **Subsampling the pixels will not change the object** ——property 3

##### 两个架构

convolution架构：针对property 1和property 2

max pooling架构：针对property 3

##### 一个理念

针对不同的application要设计符合它特性的network structure，而不是生硬套用，这就是CNN架构的设计理念：

==**应用之道，存乎一心**==