# pytorch-tips

> 本文记录pytorch中一些比较重要的知识点和实践细节

- 任何以`_`为后缀的函数都会直接修改对象本身，如`x.zero_()`、`x.detach_()`

- 生成函数：

    - `torch.rand(*sizes)`

        返回一个张量，包含从(0, 1)均匀分布中抽取的一组随机数，张量维数由`*sizes`决定

        ~~~python
        >>> torch.rand(3,2)
        tensor([[0.0053, 0.2086],
                [0.6131, 0.1812],
                [0.4234, 0.0512]])
        ~~~

    - `torch.randn(*sizes)`

        返回一个张量，包含从均值为0、方差为1的标准正态分布中抽取的一组随机数，张量维数由`*sizes`决定

        ~~~python
        >>> torch.randn(3,2)
        tensor([[ 0.2132, -0.4004],
                [ 1.4970, -0.4078],
                [-1.2865, -0.9541]])
        ~~~

    - `torch.linspace(start, end, steps=100)`

        返回一个1维张量，包含区间(start, end)中均匀间隔的steps个点，默认为100

        ~~~python
        >>> torch.linspace(1,10,19)
        tensor([ 1.0000,  1.5000,  2.0000,  2.5000,  3.0000,  3.5000,  4.0000,  4.5000,  5.0000,  5.5000,  6.0000,  6.5000,  7.0000,  7.5000,  8.0000,  8.5000,  9.0000,  9.5000, 10.0000])
        ~~~

- `tensor.view(n, d)`就等价于`np.reshape(n, d)`

- `tensor.item()`用于取出tensor中的元素，并返回纯数值，注意此时tensor必须只含一个元素

- `tensor.numpy()`用于将tensor转化成numpy，`c=torch.from_numpy(b)`用于将numpy类型的b转化成tensor类型的c

    ~~~python
    >>> a=torch.tensor([[2.],[3.]])
    >>> a
    tensor([[2.],
            [3.]])
    >>> b=a.numpy()
    >>> b
    array([[2.],
           [3.]], dtype=float32)
    >>> c=torch.from_numpy(b)
    >>> c
    tensor([[2.],
            [3.]])
    ~~~

- 当设定tensor的`requires_grad`属性为`True`时，pytorch会自动保存计算图，而`tensor.grad_fn`则保存了计算图中创造了该tensor的函数

    当对某个tensor执行`tensor.backward()`操作时，以该tensor为终点的路径都会进行自动微分，之后就可以通过`tensor.grad`获取微分值

    注意，pytorch默认不清空上一次的grad值，而是做梯度累加，因此每次执行`tensor.backward()`前都需要对计算图路径上的目标tensor做`tensor.grad.zero_()`以便清零之前的梯度值

    在神经网络中，我们通常在每个epoch中，执行梯度下降前，对优化器执行`optimizer.zero_()`操作，以便清除上一个epoch残留下的梯度值

- `tensor.detach()`可以获得与tensor相同的内容，但不包含梯度grad

- pytorch保存和提取神经网络的两种途径：

    - 保存：`torch.save(model, 'model.pkl')`

        特点：保存整个网络

        提取：`model = torch.load('model.pkl')`

        特点：读取整个神经网络，速度慢

    - 保存：`torch.save(model.state_dict(), 'model_params.pkl')`

        特点：只保存网络中的参数，速度快，占内存少

        提取：`model=Net(...)`、`model.load_state_dict(torch.load('model_params.pkl'))`

        特点：只读取参数，需要先重新建立好网络

- 在`import torchverison`的时候报错：ImportError: cannot import name 'PILLOW_VERSION'

    原因是在高版本的PIL模块中`PILLOW_VERSION`已经被修改为`__version__`

    解决方案：

    ~~~bash
    $ cd ~/anaconda3/envs/ml/lib/python3.6/site-packages/torchvision/transforms
    $ vim functional.py
    
    # from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
    from PIL import Image, ImageOps, ImageEnhance, __version__ 
    ~~~

- pytorch在很多场景下都需要处理数据类型和归一化，否则会在运行时报错

    比如cross_entropy需要input Long的数据

    input的data如果是手动处理的话一般都要设为`.type(FloatTensor)/255`

- 关于`nn.函数()`和`nn.functionals.函数()`的区别：

    - 前者是封装好的类，后者是函数
    - 前者不需要自己定义weight、bias，但后者需要
    - 前者可以直接在`Sequential()`中使用，但后者不行
    - 具体看https://www.zhihu.com/question/66782101
    - PyTorch官方推荐：具有学习参数的（例如，conv2d, linear, batch_norm)采用`nn.Xxx`方式，没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用`nn.functional.xxx`或者`nn.Xxx`方式。但关于dropout，个人强烈推荐使用`nn.Xxx`方式

- `Categorical(prob)`使用说明

    - 输入的参数prob是一个概率分布

    ~~~python
    # 生成一个从0~N-1的下标序列，其中N为prob的长度
    # 这个序列中，每个下标被抽到的概率等于prob中的对应概率
    action_dist = Categorical(prob)
    
    # 从序列中按照prob概率抽取下标
    action = action_dist.sample()
    # 返回抽取到的下标所对应的概率取对数值后的结果
    prob = action_dist.log_prob(action)
    ~~~

- `model.train()`使用说明

    - `model.train()`可以告诉模型你正处于训练模式，所以类似`dropout`、`batchnorm`等操作是会生效的
    - `model.train(mode=False)`则表示当前不处于训练模式，可以被当做稳定的用来测试的模型，则此时`dropout`、`batchnorm`等只在训练时使用的操作并不会生效

- `torch.tensor`无法直接利用类似`append`的方法添加新元素

    可以先用list或numpy处理完后再添加

    - `np.append(arr, item)`：arr为`np.array()`类型，item为一个元素
    - `np.concatenate(arr, arr2)`：将两个`np.array()`类型拼接，axis参数控制拼接方向
    - `torch.from_numpy(arr)`：将`np.array()`类型转化为`torch.tensor()`类型
    - `np.full(n, item)`：创建长度为n的数组，全部用item值填充

    ~~~python
    >>> arr = np.array([1,2,3])
    >>> np.append(arr, 666)
    array([  1,   2,   3, 666])
    >>> np.concatenate((arr, [233]))
    array([  1,   2,   3, 233])
    >>> np.concatenate((arr, np.full(3,2)), axis=0)
    array([1, 2, 3, 2, 2, 2])
    ~~~

- 自行设计损失函数时，从处理NN输出一直到计算损失函数，都**必须要保证任何与NN参数有关的中间变量都是可微的**，即`requires_grad=True`

- `torch.from_numpy()`得到的是double类型(float64)，而`torch.tensor()`得到的是float类型(float32)，因此有时候我们需要对变量最适当的数据类型转化

    `tensor.type(..Tensor)`可用于将tensor数据类型转换至“..Tensor”类型

    ~~~python
    >>> a=np.array([1,2,3])
    >>> a=torch.from_numpy(a)
    >>> a.dtype
    torch.int64
    >>> a=a.type(torch.FloatTensor)
    >>> a.dtype
    torch.float32
    >>> a=a.type(torch.DoubleTensor)
    >>> a.dtype
    torch.float64
    ~~~

- 在列表中使用`ls.append(tensor)`去叠加tensor，本质上就是形成一个tensor的参数列表

    我们在使用某些合并变形函数时，`f([tensor1, tensor2, ...])`其实就等价于`f((tensor1, tensor2, ...))`

- `torch.stack((tensor1, tensor2, ...), dim=0)`

    `torch.cat((tensor1, tensor2, ...), dim=0)`

    两种都用于矩阵合并，默认dim=0表示默认按照列方向(竖着)合并

    stack与cat的区别在于，torch.stack()函数要求输入张量的大小完全相同，得到的张量的维度会比输入的张量的大小多1，并且多出的那个维度就是拼接的维度，那个维度的大小就是输入张量的个数

    ~~~python
    # 如果我们用一个列表保存了tensor的序列，则这个列表可以直接当做stack或cat的参数，来实现tensor序列的合并操作
    >>> ls=[]
    >>> a=torch.tensor(1., requires_grad=True)
    >>> ls.append(a)
    >>> a=torch.tensor(2., requires_grad=True)
    >>> ls.append(a)
    >>> a=torch.tensor(3., requires_grad=True)
    >>> ls.append(a)
    >>> ls
    [tensor(1., requires_grad=True), tensor(2., requires_grad=True), tensor(3., requires_grad=True)]
    >>> torch.stack(ls)
    tensor([1., 2., 3.], grad_fn=<StackBackward>)
    ~~~

- pytorch计算图的释放机制

    pytoch构建的计算图是动态图，为了节约内存，所以每次一轮迭代完之后计算图就被在内存释放，所以当你想要多次**backward**的时候就会报如下错：

    *Trying to backward through the graph a second time, but the buffers have already been freed. Specify retain_graph=True when calling backward the first time.*

    解决方案：需要多次**backward**只需要在第一次反向传播时候添加一个标识：

    ~~~python
    loss.backward(retain_graph=True)
    ~~~

- `torchvision.transforms`包提供数据处理的方法，一般用于`Dataset`类的内部进行数据处理

    - 通常使用`transforms.Compose(...)`将多个`transforms`组合在一起

    - `transforms.ToPILImage()`：将数据转换为PILImage
    - `transforms.Resize((n,m))`：将PILImage的大小缩放为n×m
    - `transforms.ToTensor()`：将PILImage或numpy.ndarray类型的数据转化为`torch.FloatTensor`类型，并且自动将数据归一化到[0, 1]区间
    - `transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`：对三个通道同时进行均一化处理，令channel=(channel-mean) / std，此时原本[0, 1]的空间就映射到了[-1, 1]的空间

    ~~~python
    img_transform = torchvision.transforms.Compose(
        [
            transforms.ToPILImage(),
         	transforms.Resize((64, 64)),
         	transforms.ToTensor(),
         	transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) 
        ] 
    )
    ~~~

- `torch.utils.data.Dataset`用于包装、读取和处理数据，需要用到之前定义的`transforms`，并且要重载3个函数：

    ~~~python
    class DatasetName(torch.utils.data.Dataset):
        # 读进数据和标签(标签不是必须的)
        def __init__(self, your_data, your_label, your_transforms):
            self.data = your_data
            self.label = your_label
            self.transform = your_transforms
        
        # 返回数据的长度
        def __len__(self):
            return len(your_data)
        
        # 根据下标返回数据和标签，如果是无监督学习，则不用返回标签
        def __getitem(self, index):
            data, label =  self.your_data[index], self.your_label[index]
            data = ... # 处理数据
            data = self.transform(data) # 最后利用transform处理数据
            return data, label
    ~~~

    创建实例时需要用到负责数据处理的`transforms`

    ~~~python
    train_dataset = DatasetName(data, label, transform)
    ~~~

- `torch.utils.data.DataLoader`

    有了数据读取的方式`Dataset`和数据处理的方式`transforms`，接下来就要用它们去定义一个数据加载集`DataLoader`，它可以读取、处理数据，并按照指定的方式提供shuffle后的batch数据

    参数解释如下：

    - **dataset**：Dataset类型，从其中加载数据 
    - **batch_size**：int，可选，每个batch加载多少样本 
    - **shuffle**：bool，可选，为True时表示每个epoch都对数据进行洗牌 
    - **num_workers**：int，可选，加载数据时使用多少子进程，默认值为0，表示在主进程中加载数据
    - sampler：Sampler，可选，从数据集中采样样本的方法
    - collate_fn：callable，可选 
    - pin_memory：bool，可选 
    - drop_last：bool，可选，True表示如果最后剩下不完全的batch,丢弃，False表示不丢弃

    ~~~python
    train_loader = torch.utils.data.DataLoader(
    	dataset = train_dataset, 	# 之前准备好含有transforms的Dataset
        batch_size = 20,  			# 定义一个batch的大小
        shuffle = True,  			# 打乱数据
        num_worksers = 2			# 多线程读取数据
    )
    ~~~

- `nn.CrossEntropyLoss()`内置softmax层，因此无需在神经网络的输出层使用softmax，做分类有两种选择：

    - 使用线性无激活函数的`nn.Linear()`当输出层，并用`nn.CrossEntropyLoss()`作为损失函数
    - 使用softmax函数`F.softmax()`作为输出层的激活函数，并用`nn.MSELoss()`作为损失函数

- 使用pytorch做分类时，给`nn.CrossEntropyLoss()`==输入的是0\~n-1的一维索引值，而不是one-hot编码==！！！因此类别标签无需做one-hot编码，只需要给出0\~n-1就好了！！！

    - 索引值类型：(n, )，不能是(n, 1)，必须是一维的行向量
    - 详情看这篇[博客](https://blog.csdn.net/c2250645962/article/details/106014693)

    
