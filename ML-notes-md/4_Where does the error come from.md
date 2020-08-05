# Where does the error come from?

#### Review

之前有提到说，不同的function set，也就是不同的model，它对应的error是不同的；越复杂的model，也许performance会越差，所以今天要讨论的问题是，这个error来自什么地方

* error due to ==**bias**==
* error due to ==**variance**==

了解error的来源其实是很重要的，因为我们可以针对它挑选适当的方法来improve自己的model，提高model的准确率，而不会毫无头绪

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/estimator.png" style="width:60%;" /></center>
#### 抽样分布

##### $\widehat{y}$和$y^*$ 真值和估测值

$\widehat{y}$表示那个真正的function，而$f^*$表示这个$\widehat{f}$的估测值estimator

就好像在打靶，$\widehat{f}$是靶的中心点，收集到一些data做training以后，你会得到一个你觉得最好的function即$f^*$，这个$f^*$落在靶上的某个位置，它跟靶中心有一段距离，这段距离就是由Bias和variance决定的

bias：偏差；variance：方差  -> 实际上对应着物理实验中系统误差和随机误差的概念，假设有n组数据，每一组数据都会产生一个相应的$f^*$，此时bias表示所有$f^*$的平均落靶位置和真值靶心的距离，variance表示这些$f^*$的集中程度

##### 抽样分布的理论(概率论与数理统计)

假设独立变量为x(这里的x代表每次独立地从不同的training data里训练找到的$f^*$)，那么

总体期望$E(x)=u$ ；总体方差$Var(x)=\sigma^2$ 

###### 用样本均值$\overline{x}$估测总体期望$u$

由于我们只有有限组样本 $Sample \ N \ points:\{x^1,x^2,...,x^N\}$，故

样本均值$\overline{x}=\frac{1}{N}\sum\limits_{i=1}^{N}x^i$ ；样本均值的期望$E(\overline{x})=E(\frac{1}{N}\sum\limits_{i=1}^{N}x^i)=u$ ; 样本均值的方差$Var(\overline{x})=\frac{\sigma^2}{N}$

**样本均值 $\overline{x}$的期望是总体期望$u$**，也就是说$\overline{x}$是按概率对称地分布在总体期望$u$的两侧的；而$\overline{x}$分布的密集程度取决于N，即数据量的大小，如果N比较大，$\overline{x}$就会比较集中，如果N比较小，$\overline{x}$就会以$u$为中心分散开来

综上，==样本均值$\overline{x}$以总体期望$u$为中心对称分布，可以用来估测总体期望$u$==

###### 用样本方差$s^2$估测总体方差$\sigma^2$

由于我们只有有限组样本 $Sample \ N \ points:\{x^1,x^2,...,x^N\}$，故

样本均值$\overline{x}=\frac{1}{N}\sum\limits_{i=1}^{N}x^i$ ；样本方差$s^2=\frac{1}{N-1}\sum\limits_{i=1}^N(x^i-\overline{x})^2$ ；样本方差的期望$E(s^2)=\sigma^2$ ； 样本方差的方差$Var(s^2)=\frac{2\sigma^4}{N-1}$

**样本方差$s^2$的期望是总体方差$\sigma^2$**，而$s^2$分布的密集程度也取决于N

同理，==样本方差$s^2$以总体方差$\sigma^2$为中心对称分布，可以用来估测总体方差$\sigma^2$==

##### 回到regression的问题上来

现在我们要估测的是靶的中心$\widehat{f}$，每次collect data训练出来的$f^*$是打在靶上的某个点；产生的error取决于：

* 多次实验得到的$f^*$的期望$\overline{f}$与靶心$\widehat{f}$之间的bias——$E(f^*)$，可以形象地理解为瞄准的位置和靶心的距离的偏差
* 多次实验的$f^*$之间的variance——$Var(f^*)$，可以形象地理解为多次打在靶上的点的集中程度

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bias-variance.png" /></center>
说到这里，可能会产生一个疑惑：我们之前不就只做了一次实验吗？我们就collect了十笔data，然后training出来了一个$f^*$，然后就结束了。那怎么找很多个$f^*$呢？怎么知道它的bias和variance有多大呢？

##### $f^*$取决于model的复杂程度以及data的数量

假设这里有多个平行宇宙，每个空间里都在用10只宝可梦的data去找$f^*$，由于不同宇宙中宝可梦的data是不同的，因此即使使用的是同一个model，最终获得的$f^*$都会是不同的

于是我们做100次相同的实验，把这100次实验找出来的100条$f^*$的分布画出来

###### $f^*$的variance取决于model的复杂程度和data的数量

$f^*$的variance是由model决定的，一个简单的model在不同的training data下可以获得比较稳定分布的$f^*$，而复杂的model在不同的training data下的分布比较杂乱(如果data足够多，那复杂的model也可以得到比较稳定的分布)

如果采用比较简单的model，那么每次在不同data下的实验所得到的不同的$f^*$之间的variance是比较小的，就好像说，你在射击的时候，每次击中的位置是差不多的，就如同下图中的linear model，100次实验找出来的$f^*$都是差不多的

但是如果model比较复杂，那么每次在不同data下的实验所得到的不同的$f^*$之间的variance是比较大的，它的散布就会比较开，就如同下图中含有高次项的model，每一条$f^*$都长得不太像，并且散布得很开

> 那为什么比较复杂的model，它的散布就比较开呢？比较简单的model，它的散布就比较密集呢？

原因其实很简单，其实前面在讲regularization正规化的时候也提到了部分原因。简单的model实际上就是没有高次项的model，或者高次项的系数非常小的model，这样的model表现得相当平滑，受到不同的data的影响是比较小的

举一个很极端的例子，我们的整个model(function set)里面，就一个function：f=c，这个function只有一个常数项，因此无论training data怎么变化，从这个最简单的model里找出来的$f^*$都是一样的，它的variance就是等于0

###### $f^*$的bias只取决于model的复杂程度

bias是说，我们把所有的$f^*$平均起来得到$E(f^*)=\overline{f^*}$，这个$\overline{f^*}$与真值$\widehat{f}$有多接近

当然这里会有一个问题是说，总体的真值$\widehat{f}$我们根本就没有办法知道，因此这里只是假定了一个$\widehat{f}$

下面的图示中，**红色**线条部分代表5000次实验分别得到的$f^*$，**黑色**线条部分代表真实值$\widehat{f}$，**蓝色**线条部分代表5000次实验得到的$f^*$的平均值$\overline{f}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/5000-tests.png" style="width:60%;" /></center>
根据上图我们发现，当model比较简单的时候，每次实验得到的$f^*$之间的variance会比较小，这些$f^*$会稳定在一个范围内，但是它们的平均值$\overline{f}$距离真实值$\widehat{f}$会有比较大的偏差；而当model比较复杂的时候，每次实验得到的$f^*$之间的variance会比较大，实际体现出来就是每次重新实验得到的$f^*$都会与之前得到的有较大差距，但是这些差距较大的$f^*$的平均值$\overline{f}$却和真实值$\widehat{f}$比较接近

上图分别是含有一次项、三次项和五次项的model做了5000次实验后的结果，你会发现model越复杂，比如含有5次项的model那一幅图，每一次实验得到的$f^*$几乎是杂乱无章，遍布整幅图的；但是他们的平均值却和真实值$\widehat{f}$吻合的很好。也就是说，复杂的model，单次实验的结果是没有太大参考价值的，但是如果把考虑多次实验的结果的平均值，也许会对最终的结果有帮助

注：这里的单次实验指的是，用一组training data训练出model的一组有效参数以构成$f^*$(每次独立实验使用的training data都是不同的)

###### 因此：

* 如果是一个比较简单的model，那它有比较小的variance和比较大的bias。就像下图中左下角的打靶模型，每次实验的$f^*$都比较集中，但是他们平均起来距离靶心会有一段距离(比较适合实验次数少甚至只有单次实验的情况)
* 如果是一个比较复杂的model，每次实验找出来的$f^*$都不一样，它有比较大的variance但是却有比较小的bias。就像下图中右下角的打靶模型，每次实验的$f^*$都比较分散，但是他们平均起来的位置与靶心比较接近(比较适合多次实验的情况)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/model-bias.png" style="width:60%;" /></center>
###### 为什么会这样？

实际上我们的model就是一个function set，当你定好一个model的时候，实际上就已经定好这个function set的范围了，那个最好的function只能从这个function set里面挑出来

如果是一个简单的model，它的function set的space是比较小的，这个范围可能根本就没有包含你的target；如果这个function set没有包含target，那么不管怎么sample，平均起来永远不可能是target(这里的space指上图中左下角那个被model圈起来的空间)

如果这个model比较复杂，那么这个model所代表的function set的space是比较大的(简单的model实际上就是复杂model的子集)，那它就很有可能包含target，只是它没有办法找到那个target在哪，因为你给的training data不够，你给的training data每一次都不一样，所以他每一次找出来的$f^*$都不一样，但是如果他们是散布在这个target附近的，那平均起来，实际上就可以得到和target比较接近的位置(这里的space指上图中右下角那个被model圈起来的空间)

#### Bias vs Variance

由前面的讨论可知，比较简单的model，variance比较小，bias比较大；而比较复杂的model，bias比较小，variance比较大

##### bias和variance对error的影响

因此下图中(也就是之前我们得到的从最高项为一次项到五次项的五个model的error表现)，绿色的线代表variance造成的error，红色的线代表bias造成的error，蓝色的线代表这个model实际观测到的error

$error_{实际}=error_{variance}+error_{bias}——蓝线为红线和绿线之和$

可以发现，随着model的逐渐复杂：

* bias逐渐减小，bias所造成的error也逐渐下降，也就是打靶的时候瞄得越来越准，体现为图中的红线
* variance逐渐变大，variance所造成的error也逐渐增大，也就是虽然瞄得越来越准，但是每次射出去以后，你的误差是越来越大的，体现为图中的绿线
* 当bias和variance这两项同时被考虑的时候，得到的就是图中的蓝线，也就是实际体现出来的error的变化；实际观测到的error先是减小然后又增大，因此实际error为最小值的那个点，即为bias和variance的error之和最小的点，就是表现最好的model
* ==**如果实际error主要来自于variance很大，这个状况就是overfitting过拟合；如果实际error主要来自于bias很大，这个状况就是underfitting欠拟合**==(可以理解为，overfitting就是过分地包围了靶心所在的space，而underfitting则是还未曾包围到靶心所在的space)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bias-vs-variance.png" style="width:60%;"></center>
这就是为什么我们之前要先计算出每一个model对应的error(每一个model都有唯一对应的$f^*$，因此也有唯一对应的error)，再挑选error最小的model的原因，只有这样才能综合考虑bias和variance的影响，找到一个实际error最小的model

##### 必须要知道自己的error主要来自于哪里

###### 你现在的问题是bias大，还是variance大？

当你自己在做research的时候，你必须要搞清楚，手头上的这个model，它目前主要的error是来源于哪里；你觉得你现在的问题是bias大，还是variance大

你应该先知道这件事情，你才能知道你的future work，你要improve你的model的时候，你应该要走哪一个方向

###### 那怎么知道现在是bias大还是variance大呢？

* 如果model没有办法fit training data的examples，代表bias比较大，这时是underfitting

    形象地说，就是该model找到的$f^*$上面并没有training data的大部分样本点，如下图中的linear model，我们只是example抽样了这几个蓝色的样本点，而这个model甚至没有fit这少数几个蓝色的样本点(这几个样本点没有在$f^*$上)，代表说这个model跟正确的model是有一段差距的，所以这个时候是bias大的情况，是underfitting

* 如果model可以fit training data，在training data上得到小的error，但是在testing data上，却得到一个大的error，代表variance比较大，这时是overfitting

###### 如何针对性地处理bias大 or variance大的情况呢？

遇到bias大或variance大的时候，你其实是要用不同的方式来处理它们

1、**如果bias比较大**

bias大代表，你现在这个model里面可能根本没有包含你的target，$\widehat{f}$可能根本就不在你的function set里

对于error主要来自于bias的情况，是由于该model(function set)本来就不好，collect更多的data是没有用的，必须要从model本身出发

* redesign，重新设计你的model

    * 增加更多的features作为model的input输入变量

        比如pokemon的例子里，只考虑进化前cp值可能不够，还要考虑hp值、species种类...作为model新的input变量

    * 让model变得更复杂，增加高次项

        比如原本只是linear model，现在考虑增加二次项、三次项...

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/large-bias.png" style="width:60%;" /></center>
2、**如果variance比较大**

* 增加data
    * 如果是5次式，找100个$f^*$，每次实验我们只用10只宝可梦的数据训练model，那我们找出来的100个$f^*$的散布就会像下图一样杂乱无章；但如果每次实验我们用100只宝可梦的数据训练model，那我们找出来的100个$f^*$的分布就会像下图所示一样，非常地集中
    * 增加data是一个很有效控制variance的方法，假设你variance太大的话，collect data几乎是一个万能丹一样的东西，并且它不会伤害你的bias
    * 但是它存在一个很大的问题是，实际上并没有办法去collect更多的data
    * 如果没有办法collect更多的data，其实有一招，根据你对这个问题的理解，自己去generate更多“假的”data
        * 比如手写数字识别，因为每个人手写数字的角度都不一样，那就把所有training data里面的数字都左转15°，右转15°
        * 比如做火车的影像辨识，只有从左边开过来的火车影像资料，没有从右边开过来的火车影像资料，该怎么办？实际上可以把每张图片都左右颠倒，就generate出右边的火车数据了，这样就多了一倍data出来
        * 比如做语音辨识的时候，只有男生说的“你好”，没有女生说的“你好”，那就用男生的声音用一个变声器把它转化一下，这样男女生的声音就可以互相转化，这样data就可以多出来
        * 比如现在你只有录音室里录下的声音，但是detection实际要在真实场景下使用的，那你就去真实场景下录一些噪音加到原本的声音里，就可以generate出符合条件的data了
* Regularization(正规化)
    * 就是在loss function里面再加一个与model高次项系数相关的term，它会希望你的model里高次项的参数越小越好，也就是说希望你今天找出来的曲线越平滑越好；这个新加的term前面可以有一个weight，代表你希望你的曲线有多平滑
    * 下图中Regularization部分，左边第一幅图是没有加regularization的test；第二幅图是加了regularization后的情况，一些怪怪的、很不平滑的曲线就不会再出现，所有曲线都集中在比较平滑的区域；第三幅图是增加weight的情况，让曲线变得更平滑
    * 加了regularization以后，因为你强迫所有的曲线都要比较平滑，所以这个时候也会让你的variance变小；但regularization是可能会伤害bias的，因为它实际上调整了function set的space范围，变成它只包含那些比较平滑的曲线，这个缩小的space可能没有包含原先在更大space内的$\widehat{f}$，因此伤害了bias，所以当你做regularization的时候，需要调整regularization的weight，在variance和bias之间取得平衡

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/large-variance.png" style="width:60%;"/></center>
注：variance比较大的case，加以图例解释如下：(假设这里我们无法获得更多的data)

1、蓝色区域代表最初的情况，此时model比较复杂，function set的space范围比较大，包含了target靶心，但由于data不够，$f^*$比较分散，variance比较大

2、红色区域代表进行regularization之后的情况，此时model的function set范围被缩小成只包含平滑的曲线，space减小，variance当然也跟着变小，但这个缩小后的space实际上并没有包含原先已经包含的target靶心，因此该model的bias变大

3、橙色区域代表增大regularization的weight的情况，增大weight实际上就是放大function set的space，慢慢调整至包含target靶心，此时该model的bias变小，而相较于一开始的case，由于限定了曲线的平滑度(由weight控制平滑度的阈值)，该model的variance也比较小

实际上，通过regularization优化model的过程就是上述的1、2、3步骤，不断地调整regularization的weight，使model的bias和variance达到一个最佳平衡的状态(可以通过error来评价状态的好坏，weight需要慢慢调参)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization-illustration.png" style="width:60%;"/></center>
#### Model Selection

我们现在会遇到的问题往往是这样：我们有很多个model可以选择，还有很多参数可以调，比如regularization的weight，那通常我们是在bias和variance之间做一些trade-off权衡

我们希望找一个model，它variance够小，bias也够小，这两个合起来给我们最小的testing data的error

##### 但是以下这些事情，是你不应该做的：

你手上有training set，有testing set，接下来你想知道model1、model2、model3里面，应该选哪一个model，然后你就分别用这三个model去训练出$f_1^*,f_2^*,f_3^*$，然后把它apply到testing set上面，分别得到三个error为0.9，0.7，0.5，这里很直觉地会认为是model3最好

但是现在可能的问题是，这个testing set是你自己手上的testing set，是你自己拿来衡量model好坏的testing set，真正的testing set是你没有的；注意到你自己手上的这笔testing set，它有自己的一个bias(这里的bias跟之前提到的略有不同，可以理解为自己的testing data跟实际的testing data会有一定的偏差存在)

所以你今天那这个testing set来选择最好的model的时候，它在真正的testing set上不见得是最好的model，通常是比较差的，所以你实际得到的error是会大于你在自己的testing set上估测到的0.5

以PM2.5预测为例，提供的数据分为training set，public testing set和private testing set三部分，其中public的testing set是供你测试自己的model的，private的testing data是你暂且未知的真正测试数据，现在你的model3在public testing set上的error为0.5，已经成功beat baseline，但是在private的testing set上，你的model3也许根本就没有beat the baseline，反而是model1和model2可能会表现地更好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/model-selection.png" style="width:60%;"/></center>
##### 怎样做才是可靠的呢？

###### training data分成training set和validation set

你要做的事情是，把你的training set分成两组：

* 一组是真正拿来training model的，叫做training set(训练集)
* 另外一组不拿它来training model，而是拿它来选model，叫做validation set(验证集)

==先在training set上找出每个model最好的function $f^*$，然后用validation set来选择你的model==

也就是说，你手头上有3个model，你先把这3个model用training set训练出三个$f^*$，接下来看一下它们在validation set上的performance

假设现在model3的performance最好，那你可以直接把这个model3的结果拿来apply在testing data上

如果你担心现在把training set分成training和validation两部分，感觉training data变少的话，可以这样做：已经从validation决定model3是最好的model，那就定住model3不变(function的表达式不变)，然后用全部的data在model3上面再训练一次(使用全部的data去更新model3表达式的参数)

这个时候，如果你把这个训练好的model的$f^*$apply到public testing set上面，你可能会得到一个大于0.5的error，虽然这么做，你得到的error表面上看起来是比较大的，但是**这个时候你在public set上的error才能够真正反映你在private set上的error**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cross-validation.png" style="width:60%;"/></center>
###### 考虑真实的测试集

实际上是这样一个关系：

> training data(训练集) -> 自己的testing data(测试集) -> 实际的testing data 
> (该流程没有考虑自己的testing data的bias)

> training set(部分训练集) -> validation set(部分验证集) -> 自己的testing data(测试集) -> 实际的testing data 
> (该流程使用自己的testing data和validation来模拟testing data的bias误差，可以真实地反映出在实际的data上出现的error)

###### 真正的error

当你得到public set上的error的时候(尽管它可能会很大)，不建议回过头去重新调整model的参数，因为当你再回去重新调整什么东西的时候，你就又会把public testing set的bias给考虑进去了，这就又回到了第一种关系，即围绕着有偏差的testing data做model的优化

这样的话此时你在public set上看到的performance就没有办法反映实际在private set上的performance了，因为你的model是针对public set做过优化的，虽然public set上的error数据看起来可能会更好看，但是针对实际未知的private set，这个“优化”带来的可能是反作用，反而会使实际的error变大

当然，你也许几乎没有办法忍住不去做这件事情，在发paper的时候，有时候你会propose一个方法，那你要attach在benchmark的corpus，如果你在testing set上得到一个差的结果，你也几乎没有办法把持自己不回头去调一下你的model，你肯定不会只是写一个paper说这个方法不work这样子(滑稽

因此这里只是说，你要keep in mind，如果在那个benchmark corpus上面所看到的testing的performance，它的error，肯定是大于它在real的application上应该有的值

比如说你现在常常会听到说，在image lab的那个corpus上面，error rate都降到3%，那个是超越人类了，但是真的是这样子吗？已经有这么多人玩过这个corpus，已经有这么多人告诉你说前面这些方法都不work，他们都帮你挑过model了，你已经用“testing” data调过参数了，所以如果你把那些model真的apply到现实生活中，它的error rate肯定是大于3%的

###### 如何划分training set和validation set？

那如果training set和validation set分坏了怎么办？如果validation也有怪怪的bias，岂不是对结果很不利？那你要做下面这件事情：

==**N-flod Cross Validation**==

如果你不相信某一次分train和validation的结果的话，那你就分很多种不同的样子

比如说，如果你做3-flod的validation，意思就是你把training set分成三份，你每一次拿其中一份当做validation set，另外两份当training；分别在每个情境下都计算一下3个model的error，然后计算一下它的average error；然后你会发现在这三个情境下的average error，是model1最好

然后接下来，你就把用整个完整的training data重新训练一遍model1的参数；然后再去testing data上test

原则上是，如果你少去根据public testing set上的error调整model的话，那你在private testing set上面得到的error往往是比较接近public testing set上的error的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/n-flod-cross-validation.png" style="width:60%;"/></center>
#### 总结conclusion

1、一般来说，error是bias和variance共同作用的结果

2、model比较简单和比较复杂的情况：

* 当model比较简单的时候，variance比较小，bias比较大，此时$f^*$会比较集中，但是function set可能并没有包含真实值$\widehat{f}$；此时model受bias影响较大
* 当model比较复杂的时候，bias比较小，variance比较大，此时function set会包含真实值$\widehat{f}$，但是$f^*$会比较分散；此时model受variance影响较大

3、区分bias大 or variance大的情况

* 如果连采样的样本点都没有大部分在model训练出来的$f^*$上，说明这个model太简单，bias比较大，是欠拟合

* 如果样本点基本都在model训练出来的$f^*$上，但是testing data上测试得到的error很大，说明这个model太复杂，variance比较大，是过拟合

4、bias大 or variance大的情况下该如何处理

* 当bias比较大时，需要做的是重新设计model，包括考虑添加新的input变量，考虑给model添加高次项；然后对每一个model对应的$f^*$计算出error，选择error值最小的model(随model变复杂，bias会减小，variance会增加，因此这里分别计算error，取两者平衡点)

* 当variance比较大时，一个很好的办法是增加data(可以凭借经验自己generate data)，当data数量足够时，得到的$f^*$实际上是比较集中的；如果现实中没有办法collect更多的data，那么就采用regularization正规化的方法，以曲线的平滑度为条件控制function set的范围，用weight控制平滑度阈值，使得最终的model既包含$\widehat{f}$，variance又不会太大

5、如何选择model

* 选择model的时候呢，我们手头上的testing data与真实的testing data之间是存在偏差的，因此我们要将training data分成training set和validation set两部分，经过validation挑选出来的model再用全部的training data训练一遍参数，最后用testing data去测试error，这样得到的error是模拟过testing bias的error，与实际情况下的error会比较符合