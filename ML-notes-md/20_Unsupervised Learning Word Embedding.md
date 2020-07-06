### Unsupervised Learning: Word Embedding

> 本文介绍NLP中词嵌入(Word Embedding)相关的基本知识，基于降维思想提供了count-based和prediction-based两种方法，并介绍了该思想在机器问答、机器翻译、图像分类、文档嵌入等方面的应用

#### Introduction

词嵌入(word embedding)是降维算法(Dimension Reduction)的典型应用

那如何用vector来表示一个word呢？

##### 1-of-N Encoding

最传统的做法是1-of-N Encoding，假设这个vector的维数就等于世界上所有单词的数目，那么对每一个单词来说，只需要某一维为1，其余都是0即可；但这会导致任意两个vector都是不一样的，你无法建立起同类word之间的联系

##### Word Class

还可以把有同样性质的word进行聚类(clustering)，划分成多个class，然后用word所属的class来表示这个word，但光做clustering是不够的，不同class之间关联依旧无法被有效地表达出来

##### Word Embedding

词嵌入(Word Embedding)把每一个word都投影到高维空间上，当然这个空间的维度要远比1-of-N Encoding的维度低，假如后者有10w维，那前者只需要50\~100维就够了，这实际上也是Dimension Reduction的过程

类似语义(semantic)的词汇，在这个word embedding的投影空间上是比较接近的，而且该空间里的每一维都可能有特殊的含义

假设词嵌入的投影空间如下图所示，则横轴代表了生物与其它东西之间的区别，而纵轴则代表了会动的东西与静止的东西之间的差别

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we.png" width="60%"/></center>

word embedding是一个无监督的方法(unsupervised approach)，只要让机器阅读大量的文章，它就可以知道每一个词汇embedding之后的特征向量应该长什么样子

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we2.png" width="60%"/></center>

我们的任务就是训练一个neural network，input是词汇，output则是它所对应的word embedding vector，实际训练的时候我们只有data的input，该如何解这类问题呢？

之前提到过一种基于神经网络的降维方法，Auto-encoder，就是训练一个model，让它的输入等于输出，取出中间的某个隐藏层就是降维的结果，自编码的本质就是通过自我压缩和解压的过程来寻找各个维度之间的相关信息；但word embedding这个问题是不能用Auto-encoder来解的，因为输入的向量通常是1-of-N编码，各维无关，很难通过自编码的过程提取出什么有用信息

#### Word Embedding

##### basic idea

基本精神就是，每一个词汇的含义都可以根据它的上下文来得到

比如机器在两个不同的地方阅读到了“马英九520宣誓就职”、“蔡英文520宣誓就职”，它就会发现“马英九”和“蔡英文”前后都有类似的文字内容，于是机器就可以推测“马英九”和“蔡英文”这两个词汇代表了可能有同样地位的东西，即使它并不知道这两个词汇是人名

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we3.png" width="60%"/></center>

怎么用这个思想来找出word embedding的vector呢？有两种做法：

- Count based
- Prediction based

#### Count based

假如$w_i$和$w_j$这两个词汇常常在同一篇文章中出现(co-occur)，它们的word vector分别用$V(w_i)$和$V(w_j)$来表示，则$V(w_i)$和$V(w_j)$会比较接近

假设$N_{i,j}$是$w_i$和$w_j$这两个词汇在相同文章里同时出现的次数，我们希望它与$V(w_i)\cdot V(w_j)$的内积越接近越好，这个思想和之前的文章中提到的矩阵分解(matrix factorization)的思想其实是一样的

这种方法有一个很代表性的例子是[Glove Vector](http://nlp.stanford.edu/projects/glove/)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/count-based.png" width="60%"/></center>

#### Prediction based

##### how to do perdition 

给定一个sentence，我们要训练一个神经网络，它要做的就是根据当前的word $w_{i-1}$，来预测下一个可能出现的word $w_i$是什么 

假设我们使用1-of-N encoding把$w_{i-1}$表示成feature vector，它作为neural network的input，output的维数和input相等，只不过每一维都是小数，代表在1-of-N编码中该维为1其余维为0所对应的word会是下一个word $w_i$的概率

把第一个hidden layer的input $z_1,z_2,...$拿出来，它们所组成的$Z$就是word的另一种表示方式，当我们input不同的词汇，向量$Z$就会发生变化

也就是说，第一层hidden layer的维数可以由我们决定，而它的input又唯一确定了一个word，因此提取出第一层hidden layer的input，实际上就得到了一组可以自定义维数的Word Embedding的向量

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb.png" width="60%"/></center>

##### Why prediction works

prediction-based方法是如何体现根据词汇的上下文来了解该词汇的含义这件事呢？

假设在两篇文章中，“蔡英文”和“马英九”代表$w_{i-1}$，“宣誓就职”代表$w_i$，我们希望对神经网络输入“蔡英文”或“马英九”这两个词汇，输出的vector中对应“宣誓就职”词汇的那个维度的概率值是高的

为了使这两个不同的input通过NN能得到相同的output，就必须在进入hidden layer之前，就通过weight的转换将这两个input vector投影到位置相近的低维空间上

也就是说，尽管两个input vector作为1-of-N编码看起来完全不同，但经过参数的转换，将两者都降维到某一个空间中，在这个空间里，经过转换后的new vector 1和vector 2是非常接近的，因此它们同时进入一系列的hidden layer，最终输出时得到的output是相同的

因此，词汇上下文的联系就自动被考虑在这个prediction model里面

总结一下，对1-of-N编码进行Word Embedding降维的结果就是神经网络模型第一层hidden layer的输入向量$\left [ \begin{matrix} z_1\ z_2\ ... \end{matrix} \right ]^T$，该向量同时也考虑了上下文词汇的关联，我们可以通过控制第一层hidden layer的大小从而控制目标降维空间的维数

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb2.png" width="60%"/></center>

##### Sharing Parameters

你可能会觉得通过当前词汇预测下一个词汇这个约束太弱了，由于不同词汇的搭配千千万万，即便是人也无法准确地给出下一个词汇具体是什么

你可以扩展这个问题，使用10个及以上的词汇去预测下一个词汇，可以帮助得到较好的结果

这里用2个词汇举例，如果是一般是神经网络，我们直接把$w_{i-2}$和$w_{i-1}$这两个vector拼接成一个更长的vector作为input即可

但实际上，我们希望和$w_{i-2}$相连的weight与和$w_{i-1}$相连的weight是tight在一起的，简单来说就是$w_{i-2}$与$w_{i-1}$的相同dimension对应到第一层hidden layer相同neuron之间的连线拥有相同的weight，在下图中，用同样的颜色标注相同的weight：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb3.png" width="60%"/></center>

如果我们不这么做，那把同一个word放在$w_{i-2}$的位置和放在$w_{i-1}$的位置，得到的Embedding结果是会不一样的，把两组weight设置成相同，可以使$w_{i-2}$与$w_{i-1}$的相对位置不会对结果产生影响

除此之外，这么做还可以通过共享参数的方式有效地减少参数量，不会由于input的word数量增加而导致参数量剧增

##### Formulation

假设$w_{i-2}$的1-of-N编码为$x_{i-2}$，$w_{i-1}$的1-of-N编码为$x_{i-1}$，维数均为$|V|$，表示数据中的words总数

hidden layer的input为向量$z$，长度为$|Z|$，表示降维后的维数
$$
z=W_1 x_{i-2}+W_2 x_{i-1}
$$
其中$W_1$和$W_2$都是$|Z|×|V|$维的weight matrix，它由$|Z|$组$|V|$维的向量构成，第一组$|V|$维向量与$|V|$维的$x_{i-2}$相乘得到$z_1$，第二组$|V|$维向量与$|V|$维的$x_{i-2}$相乘得到$z_2$，...，依次类推

我们强迫让$W_1=W_2=W$，此时$z=W(x_{i-2}+x_{i-1})$

因此，只要我们得到了这组参数$W$，就可以与1-of-N编码$x$相乘得到word embedding的结果$z$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb4.png" width="60%"/></center>

##### In Practice

那在实际操作上，我们如何保证$W_1$和$W_2$一样呢？

以下图中的$w_i$和$w_j$为例，我们希望它们的weight是一样的：

- 首先在训练的时候就要给它们一样的初始值

- 然后分别计算loss function $C$对$w_i$和$w_j$的偏微分，并对其进行更新
    $$
    w_i=w_i-\eta \frac{\partial C}{\partial w_i}\\
    w_j=w_j-\eta \frac{\partial C}{\partial w_j}
    $$
    这个时候你就会发现，$C$对$w_i$和$w_j$的偏微分是不一样的，这意味着即使给了$w_i$和$w_j$相同的初始值，更新过一次之后它们的值也会变得不一样，因此我们必须保证两者的更新过程是一致的，即：
    $$
    w_i=w_i-\eta \frac{\partial C}{\partial w_i}-\eta \frac{\partial C}{\partial w_j}\\
    w_j=w_j-\eta \frac{\partial C}{\partial w_j}-\eta \frac{\partial C}{\partial w_i}
    $$

- 这个时候，我们就保证了$w_i$和$w_j$始终相等：
    - $w_i$和$w_j$的初始值相同
    - $w_i$和$w_j$的更新过程相同

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb5.png" width="60%"/></center>

如何去训练这个神经网络呢？注意到这个NN完全是unsupervised，你只需要上网爬一下文章数据直接喂给它即可

比如喂给NN的input是“潮水”和“退了”，希望它的output是“就”，之前提到这个NN的输出是一个由概率组成的vector，而目标“就”是只有某一维为1的1-of-N编码，我们希望minimize它们之间的cross entropy，也就是使得输出的那个vector在“就”所对应的那一维上概率最高

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb6.png" width="60%"/></center>

##### Various Architectures

除了上面的基本形态，Prediction-based方法还可以有多种变形

- CBOW(Continuous bag of word model)

    拿前后的词汇去预测中间的词汇

- Skip-gram

    拿中间的词汇去预测前后的词汇

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb7.png" width="60%"/></center>

##### others

尽管word vector是deep learning的一个应用，但这个neural network其实并不是deep的，它就只有一个linear的hidden layer

我们把1-of-N编码输入给神经网络，经过weight的转换得到Word Embedding，再通过第一层hidden layer就可以直接得到输出

其实过去有很多人使用过deep model，但这个task不用deep就可以实现，这样做既可以减少运算量，跑大量的data，又可以节省下训练的时间(deep model很可能需要长达好几天的训练时间)

#### Application

##### Subtraction

*机器问答*

从得到的word vector里，我们可以发现一些原本并不知道的word与word之间的关系

把word vector两两相减，再投影到下图中的二维平面上，如果某两个word之间有类似包含于的相同关系，它们就会被投影到同一块区域

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we4.png" width="60%"/></center>

利用这个概念，我们可以做一些简单的推论：

- 在word vector的特征上，$V(Rome)-V(Italy)≈V(Berlin)-V(Germany)$

- 此时如果有人问“罗马之于意大利等于柏林之于？”，那机器就可以回答这个问题

    因为德国的vector会很接近于“柏林的vector-罗马的vector+意大利的vector”，因此机器只需要计算$V(Berlin)-V(Rome)+V(Italy)$，然后选取与这个结果最接近的vector即可

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we5.png" width="60%"/></center>

##### Multi-lingual Embedding

*机器翻译*

此外，word vector还可以建立起不同语言之间的联系

如果你要用上述方法分别训练一个英文的语料库(corpus)和中文的语料库，你会发现两者的word vector之间是没有任何关系的，因为Word Embedding只体现了上下文的关系，如果你的文章没有把中英文混合在一起使用，机器就没有办法判断中英文词汇之间的关系

但是，如果你知道某些中文词汇和英文词汇的对应关系，你可以先分别获取它们的word vector，然后再去训练一个模型，把具有相同含义的中英文词汇投影到新空间上的同一个点

接下来遇到未知的新词汇，无论是中文还是英文，你都可以采用同样的方式将其投影到新空间，就可以自动做到类似翻译的效果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we6.png" width="60%"/></center>

参考文献：*Bilingual Word Embeddings for Phrase-Based Machine Translation, Will Zou, Richard Socher, Daniel Cer and Christopher Manning, EMNLP, 2013*

##### Multi-domain Embedding

*图像分类*

这个做法不只局限于文字的应用，你也可以对文字+图像做Embedding

假设你已经得到horse、cat和dog这些**词汇**的vector在空间上的分布情况，你就可以去训练一个模型，把一些已知的horse、cat和dog**图片**去投影到和对应词汇相同的空间区域上

比如对模型输入一张图像，使之输出一个跟word vector具有相同维数的vector，使dog图像的映射向量就散布在dog词汇向量的周围，horse图像的映射向量就散布在horse词汇向量的周围...

训练好这个模型之后，输入新的未知图像，根据投影之后的位置所对应的word vector，就可以判断它所属的类别

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we7.png" width="60%"/></center>

我们知道在做图像分类的时候，很多情况下都是事先定好要分为哪几个具体的类别，再用这几个类别的图像去训练模型，由于我们无法在训练的时候穷尽所有类别的图像，因此在实际应用的时候一旦遇到属于未知类别的图像，这个模型就无能为力了

而使用image+word Embedding的方法，就算输入的图像类别在训练时没有被遇到过，比如上图中的cat，但如果这张图像能够投影到cat的word vector的附近，根据词汇向量与图像向量的对应关系，你自然就可以知道这张图像叫做cat

##### Document Embedding

*文档嵌入*

除了Word Embedding，我们还可以对Document做Embedding

最简单的方法是把document变成bag-of-word，然后用Auto-encoder就可以得到该文档的语义嵌入(Semantic Embedding)，但光这么做是不够的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/se.png" width="60%"/></center>

词汇的顺序代表了很重要的含义，两句词汇相同但语序不同的话可能会有完全不同的含义，比如

- 白血球消灭了传染病——正面语义
- 传染病消灭了白血球——负面语义

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/se2.png" width="60%"/></center>

想要解决这个问题，具体可以参考下面的几种处理方法：

- **Paragraph Vector**: *Le, Quoc, and Tomas Mikolov. "Distributed Representations of Sentences and Documents.“ ICML, 2014*
- **Seq2seq Auto-encoder**: *Li, Jiwei, Minh-Thang Luong, and Dan Jurafsky. "A hierarchical neural autoencoder for paragraphs and documents." arXiv preprint, 2015*
- **Skip Thought**: *Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler, “Skip-Thought Vectors” arXiv preprint, 2015.*

关于**word2vec**，可以参考博客：http://blog.csdn.net/itplus/article/details/37969519