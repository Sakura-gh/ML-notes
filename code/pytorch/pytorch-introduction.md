# pytorch tutorial

> 本文仅介绍一些pytorch的入门知识，更多内容详见官方文档：`https://pytorch.org/`

- 一些常用的基础库

    ~~~python
    # pytorch库
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # 可视化库
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    
    # numpy
    import numpy as np
    ~~~

- `torch.tensor()`和`np.array()`用法是很像的，但tensor会自动微分

- 基本操作：

    ~~~python
    # 初始化
    x_numpy = np.array([0.1, 0.2, 0.3])
    x_torch = torch.tensor([0.1, 0.2, 0.3])
    y_numpy = np.array([3, 4, 5.])
    y_torch = torch.tensor([3, 4, 5.])
    
    # 加减操作
    >>> x_torch + y_torch
    tensor([3.1000, 4.2000, 5.3000])
    >>> x_torch - y_torch
    tensor([-2.9000, -3.8000, -4.7000])
    
    # 求norm
    >>> np.linalg.norm(x_numpy)
    0.37416573867739417
    >>> torch.norm(x_torch)
    tensor(0.3742)
    
    # 求均值mean
    >>> np.mean(x_numpy)
    0.20000000000000004
    >>> torch.mean(x_torch)
    tensor(0.2000)
    
    ~~~

- `torch.view()` vs `np.reshape()`

    两者都是用于改变矩阵的形状

    ~~~python
    # 假设对MNIST的图像数据进行处理
    >>> N, C, W, H = 10000, 3, 28, 28
    >>> X = torch.randn((N, C, W, H))
    >>> X.shape
    torch.Size([10000, 3, 28, 28])
    >>> X.view(N, C, 784).shape
    torch.Size([10000, 3, 784])
    >>> X.view(-1, C, 784).shape # -1的好处是，可以自动抓取X.shape中对应维的值，懒人必备
    torch.Size([10000, 3, 784])
    ~~~

- 广播语义(Broadcasting semantics)

    tensor参数可以自动扩展为相同的大小，而不需要复制数据，但须遵守下列前提：

    - 每个 tensor 至少有一维
    - 遍历所有的维度，从尾部维度开始，每个对应的维度大小**要么相同，要么其中一个是 1，要么其中一个不存在**

    ~~~python
    # 1.相同维度，一定可以 broadcasting
    x=torch.empty(5,7,3)
    y=torch.empty(5,7,3)
    
    # 2.x没有符合“至少有一个维度”，所以不可以 broadcasting
    x=torch.empty((0,))
    y=torch.empty(2,2)
    >>> (x+y).shape
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    RuntimeError: The size of tensor a (0) must match the size of tensor b (2) at non-singleton dimension 1
    
    
    # 3.按照尾部维度对齐
    # x 和 y 是 broadcastable
    # 1st 尾部维度: 都为 1
    # 2nd 尾部维度: y 为 1
    # 3rd 尾部维度: x 和 y 相同
    # 4th 尾部维度: y 维度不存在    
    x=torch.empty(5,3,4,1)
    y=torch.empty(  3,1,1)
    >>> (x+y).shape
    torch.Size([5, 3, 4, 1])
    
    # 4.x和y不能 broadcasting, 因为尾3维度 2 != 3
    x=torch.empty(5,2,4,1)
    y=torch.empty(  3,1,1)
    >>> (x+y).shape
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1
    ~~~

- 计算图(computation graphs)

    在我们做加减乘除运算时，只要其中一项需要做gradient，pytorch就会自动建立一张计算图

    ~~~python
    # 我们需要设置requires_grad=True使pytorch知道要保存计算图的存在
    >>> a=torch.tensor(2.0, requires_grad=True)
    >>> b=torch.tensor(1.0, requires_grad=True)
    >>> c=a+b
    >>> d=b+1
    >>> e=c*d
    >>> c
    tensor(3., grad_fn=<AddBackward0>)
    >>> d
    tensor(2., grad_fn=<AddBackward0>)
    >>> e
    tensor(6., grad_fn=<MulBackward0>)
    
    ~~~

    ![tree-img](https://colah.github.io/posts/2015-08-Backprop/img/tree-eval.png)

- CUDA semantics

    使用`torch.device("cpu")`和`torch.device("cuda")`，并使用`data.to()`将数据在cpu和gpu上切换

    ~~~python
    cpu=torch.device('cpu')
    gpu=torch.device('cuda')
    
    x=torch.rand(10)
    x=x.to(gpu) # x切换到gpu上
    x=x.to(cpu) # x切换到cpu上
    ~~~

- pytorch用于自动微分

    使用`y.backward()`就可以自动计算出计算图中$y$对以$y$为终点的路径上所有变量$x^i$的gradient $\frac{\partial y}{\partial x^i}$

    例1：$f(x)=(x-2)^2,f'(x)=2(x-2)$

    ~~~python
    >>> def f(x):
    ... 	return (x-2)**2
    >>> x=torch.tensor([1.0], requires_grad=True) # 需要先建立计算图后续才能微分
    >>> y=f(x)
    >>> y.backward() # 计算图上以y为终点的路径上的所有变量都自动微分
    >>> x.grad
    tensor([-2.])
    ~~~

    例2：$w=[w_1,w_2]^T,g(w)=2w_1w_2+w_2\cos(w_1)$

    ~~~python
    >>> def g(w):
    ...     return 2*w[0]*w[1]+w[1]*torch.cos(w[0])
    >>> w=torch.tensor([np.pi, 1], requires_grad=True)
    >>> z=g(w)
    >>> z.backward()
    >>> w.grad
    tensor([2.0000, 5.2832]) # 计算出z对w[0]和w[1]的微分
    ~~~

- 使用梯度gradient

    这里依旧使用$f(x)=(x-2)^2,f'(x)=2(x-2)$，用pytorch实现梯度下降法求使$f(x)$最小的$x$

    - 注意，每次迭代时梯度都要清零，否则梯度将会默认累加

    - `tensor.item()`用于取出tensor中的纯数值，一般适用于tensor只包含一个元素的情况，多个元素可采用`tensor.tolist()`转化为列表

    - `x.grad.zero_()`用于梯度清零

    - `x.detach_()`用于切断计算图

        比如我们对y进行detach_()，就把x->y->z切成两部分：x和y->z，x就无法接受到后面传过来的梯度

    - `x.detach()`和`x.data`都是用于获取`x`这个tensor的，它们与原先的`x`这个tensor共享数据内存，因此会对`x`的值进行修改；但作为复制品，它们本身是不能自动微分的，即和它们相关的操作并不会被记录在计算图中，require s_grad = False

        两者区别在于：由于对`x.data`的修改导致原先tensor值的改变，在后续的反向传播中，用到错误的数据并不会报错；而由于对 `x.detach()`的修改导致原先tensor值的改变，在后续的反向传播中，发现原数据被修改过会报错，因而更加安全

        因此，`x.detach()`和`x.data`一般用于读取tensor数据而不用于修改，`x.detach()`更安全

    ~~~python
    >>> def f(x):
    ... 	return (x-2)**2
    >>> x=torch.tensor([5.0], requires_grad=True) # 随机定义x的初值值
    >>> lr=0.25 # 定义学习率
    >>> for i in range(15): # 做gd
    ...     y=f(x)
    ...     y.backward()
    ...     print("x=%f\t f(x)=%f\t f'(x)=%f"%(x.item(), y.item(), x.grad.item()))
    ...     x.data=x.data-lr*x.grad # 不用x.data会出问题
    ...     x.grad.detach_()
    ...     x.grad.zero_()
    ... 
    x=5.000000	 f(x)=9.000000	 f'(x)=6.000000
    x=3.500000	 f(x)=2.250000	 f'(x)=3.000000
    x=2.750000	 f(x)=0.562500	 f'(x)=1.500000
    x=2.375000	 f(x)=0.140625	 f'(x)=0.750000
    x=2.187500	 f(x)=0.035156	 f'(x)=0.375000
    x=2.093750	 f(x)=0.008789	 f'(x)=0.187500
    x=2.046875	 f(x)=0.002197	 f'(x)=0.093750
    x=2.023438	 f(x)=0.000549	 f'(x)=0.046875
    x=2.011719	 f(x)=0.000137	 f'(x)=0.023438
    x=2.005859	 f(x)=0.000034	 f'(x)=0.011719
    x=2.002930	 f(x)=0.000009	 f'(x)=0.005859
    x=2.001465	 f(x)=0.000002	 f'(x)=0.002930
    x=2.000732	 f(x)=0.000001	 f'(x)=0.001465
    x=2.000366	 f(x)=0.000000	 f'(x)=0.000732
    x=2.000183	 f(x)=0.000000	 f'(x)=0.000366
    
    ~~~

- Linear Regression

    注：矩阵相乘分为：

    - 对应点直接相乘，可以使用`x.mul(y)`或直接使用`x*y`
    - 矩阵相乘，可以使用`x.mm(y)`或直接使用`x@y`

    这里使用RSS作为损失函数：$$\nabla*_w \mathcal{L}_*{RSS}(w; X) = \nabla_w\frac{1}{n} ||y - Xw||_2^2 = -\frac{2}{n}X^T(y-Xw)$$

    1.准备数据集

    其中true_w是正确的参数w，y是通过x与w相乘并加上一定噪声获得的

    ~~~python
    d=2
    n=50
    x=torch.randn(n,d)
    true_w=torch.tensor([[-1.0],[2.0]])
    y=x@true_w+torch.randn(n,1)*0.1
    ~~~

    2.创建线性回归模型和损失函数

    ~~~python
    def model(x, w): 
        return x@w 
                                                                                   
    def rss(y, h_hat): 
        return torch.norm(y-h_hat)**2/n 
    ~~~

    3.使用梯度进行梯度下降法线性回归

    随机初始化一组参数w，然后做gradient descent

    ~~~python
    w=torch.tensor([[1.],[0]], requires_grad=True)
    
    for i in range(20): 
        y_hat=model(x, w) 
        loss=rss(y, y_hat) 
        loss.backward() 
        print('loss = %f'%loss.item(), \\
              "\tw =",w.tolist()) 
        w.data=w.data-lr*w.grad 
        w.grad.detach_() 
        w.grad.zero_() 
        
    loss = 6.877078 	w = [[1.0], [0.0]]
    loss = 4.700166 	w = [[0.6123040914535522], [0.29769018292427063]]
    loss = 3.225300 	w = [[0.3017166256904602], [0.5531282424926758]]
    loss = 2.221793 	w = [[0.052597105503082275], [0.7719148397445679]]
    loss = 1.536276 	w = [[-0.1474691480398178], [0.9590051174163818]]
    loss = 1.066256 	w = [[-0.3083454966545105], [1.1187583208084106]]
    loss = 0.742897 	w = [[-0.4378759264945984], [1.2549899816513062]]
    loss = 0.519751 	w = [[-0.5423045754432678], [1.371025800704956]]
    loss = 0.365329 	w = [[-0.6266074180603027], [1.4697538614273071]]
    loss = 0.258197 	w = [[-0.694753885269165], [1.5536739826202393]]
    loss = 0.183704 	w = [[-0.7499141693115234], [1.6249440908432007]]
    loss = 0.131802 	w = [[-0.7946230173110962], [1.6854223012924194]]
    loss = 0.095576 	w = [[-0.8309094309806824], [1.7367050647735596]]
    loss = 0.070251 	w = [[-0.8603994846343994], [1.7801612615585327]]
    loss = 0.052521 	w = [[-0.8843981027603149], [1.8169628381729126]]
    loss = 0.040094 	w = [[-0.9039536118507385], [1.8481112718582153]]
    loss = 0.031374 	w = [[-0.9199094772338867], [1.8744614124298096]]
    loss = 0.025249 	w = [[-0.9329451322555542], [1.8967417478561401]]
    loss = 0.020944 	w = [[-0.9436085224151611], [1.9155727624893188]]
    loss = 0.017916 	w = [[-0.9523422122001648], [1.931482195854187]]
    
    # 最终得到的w与[[-1],[2]]比较接近，由于对y有加random noise，因此不会相等
    ~~~

- torch.nn.Module

    1.`nn.Linear(in_features, out_features)`用于做线性变换，把原先in维的数据变换成out维

    ~~~python
    d_in=3              
    d_out=4             
    linear_module=nn.Linear(d_in,d_out)
    example_tensor=torch.tensor([[1.,2,3],[4,5,6]])
    transformed=linear_module(example_tensor)                                                             
    transformed         
    tensor([[ 0.2928, -0.7038, -0.3511,  0.5469],
            [ 0.9142, -0.6990, -1.9994,  2.6710]], grad_fn=<AddmmBackward>)
    
    linear_module.weight 
    Parameter containing:
    tensor([[ 0.1121,  0.4391, -0.3441],
            [ 0.1174,  0.2917, -0.4075],
            [-0.4640, -0.2195,  0.1340],
            [ 0.5599,  0.4045, -0.2564]], requires_grad=True)
    
    linear_module.bias
    Parameter containing:
    tensor([ 0.3347, -0.1822,  0.1498, -0.0529], requires_grad=True)
    ~~~

- 激活函数(Activation function)

    使用`nn.激活函数名()`即可

    ~~~python
    activation_f=nn.ReLU()
    tensor=torch.tensor([-1., 1., 0.])
    activated_tensor=activation_f(tensor)
    
    activated_tensor
    tensor([0., 1., 0.])
    ~~~

- 序列(Sequential)

    1.提供连续操作的简化描述方式

    ~~~python
    d_in=3
    d_hidden=4
    d_out=1
    
    model=nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.Tanh(),
        nn.Linear(d_hidden, d_out),
        nn.Sigmoid()
    )
    
    tensor_input=torch.tensor([[1., 2, 3],[4,5,6]])
    transformed=model(tensor_input)
    print(transformed)
    ~~~

    上面的式子等价于

    ~~~python
    d_in=3
    d_hidden=4
    d_out=1
    
    linear=nn.Linear(d_in, d_hidden)
    tanh=nn.Tanh()
    linear2=nn.Linear(d_hidden, d_out)
    sigmoid=nn.Sigmoid()
    
    tensor_input=torch.tensor([[1., 2, 3],[4,5,6]])
    tensor_input=linear(tensor_input)
    tensor_input=tanh(tensor_input)
    tensor_input=linear2(tensor_input)
    transformed=sigmoid(tensor_input)
    
    print(transformed)
    ~~~

    2.获取参数

    利用`model.parameters()`，返回每一层的参数

    ~~~python
    params=model.parameters()
    print(params)
    for param in params:
        print(param,'\n')
        
    <generator object Module.parameters at 0x7f0ebbaa4e08>
    Parameter containing:
    tensor([[-0.5639, -0.3145,  0.1095],
            [ 0.1118, -0.1313, -0.0237],
            [ 0.3095,  0.4493, -0.1389],
            [ 0.0412, -0.1165,  0.1050]], requires_grad=True) 
    
    Parameter containing:
    tensor([ 0.0936,  0.3826, -0.4355,  0.4673], requires_grad=True) 
    
    Parameter containing:
    tensor([[-0.2345, -0.2988,  0.2507,  0.1359]], requires_grad=True) 
    
    Parameter containing:
    tensor([-0.2972], requires_grad=True) 
    ~~~

- 损失函数(Loss Function)

    直接创建nn中loss function的对象，喂给它input和output即可获得loss

    ~~~python
    mse_loss=nn.MSELoss()
    output=torch.tensor([0., 0, 0])
    target=torch.tensor([1., 0, -1])
    
    loss=mse_loss(output, target)
    print(loss)
    tensor(0.6667)
    ~~~

- 使用优化器(Optimizer)搭建一个完整的神经网络

    下面举了几个简单的例子：

    ~~~python
    # create a simple model
    model = nn.Linear(1, 1)
    
    # create loss function
    mse_loss = nn.MSELoss()
    
    # create optimizer
    # 必须要把model的参数喂给optim，lr可以不写，会有默认参数值
    optim = torch.optim.SGD(model.parameters(), lr=1e-2) 
    
    # create a simple dataset
    x_simple = torch.tensor([[1.]])
    y_simple = torch.tensor([[2.]])
    
    # training model
    # 计算y_hat 和 loss
    y_hat = model(x_simple)
    print('model params before gd:', model.weight, '\n') # 原始参数
    loss = mse_loss(y_hat, y_simple)
    
    # 梯度清零->计算梯度->梯度下降
    # 做Backpropagation之前先将梯度清零，防止与旧值叠加
    optim.zero_grad() # 梯度清零
    loss.backward() # 反向传播，计算梯度
    optim.step() # 梯度下降
    
    # 输出结果
    print('model params after gd:', model.weight, '\n') # 梯度下降后的参数
    
    model params before gd: Parameter containing:
    tensor([[0.9044]], requires_grad=True) 
    
    model params after gd: Parameter containing:
    tensor([[0.9403]], requires_grad=True) 
    ~~~

    - 使用pytorch进行线性回归

    ~~~python
    # 准备数据集
    d = 2
    n = 50
    x = torch.randn(n, d)
    true_w = torch.tensor([[-1.0], [2.0]])
    y = x @ true_w
    
    # create a new model
    model = nn.Linear(d, 1)
    
    # create loss function
    mse_loss = nn.MSELoss()
    
    # create optimizer
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # training model
    for i in range(20):
        # 计算y_hat和loss
        y_hat = model(x)
        loss = mse_loss(y_hat, y)
        # 梯度清零->计算梯度->梯度下降
        optim.zero_grad()
        loss.backward()
        optim.step()
        # 输出中间结果
        print('loss={:.2f}\t w={}'.format(loss.item(),\\
                                          model.weight.tolist()))
      
    loss=5.53	 w=[[0.3132866621017456, 0.5566818714141846]]
    loss=3.67	 w=[[0.07761594653129578, 0.8357649445533752]]
    loss=2.43	 w=[[-0.11556379497051239, 1.0606088638305664]]
    loss=1.61	 w=[[-0.27393245697021484, 1.2418372631072998]]
    loss=1.07	 w=[[-0.4037805199623108, 1.3879727125167847]]
    loss=0.71	 w=[[-0.5102603435516357, 1.505856990814209]]
    loss=0.47	 w=[[-0.597592294216156, 1.6009858846664429]]
    loss=0.31	 w=[[-0.6692330837249756, 1.6777770519256592]]
    loss=0.21	 w=[[-0.7280142307281494, 1.7397832870483398]]
    loss=0.14	 w=[[-0.776255190372467, 1.7898640632629395]]
    loss=0.09	 w=[[-0.8158559203147888, 1.8303215503692627]]
    loss=0.06	 w=[[-0.8483729362487793, 1.8630106449127197]]
    loss=0.04	 w=[[-0.875081479549408, 1.889426350593567]]
    loss=0.03	 w=[[-0.8970263600349426, 1.9107744693756104]]
    loss=0.02	 w=[[-0.9150636792182922, 1.928027868270874]]
    loss=0.01	 w=[[-0.9298950433731079, 1.941971778869629]]
    loss=0.01	 w=[[-0.9420954585075378, 1.9532402753829956]]
    loss=0.01	 w=[[-0.952136218547821, 1.9623454809188843]]
    loss=0.00	 w=[[-0.9604036808013916, 1.9697014093399048]]
    loss=0.00	 w=[[-0.9672147035598755, 1.9756425619125366]]
    ~~~

- 使用pytorch搭建简单的神经网络，去fit函数

    1.创建数据集原始点

    ~~~python
    # 创建原始数据集(x, y)
    d = 1
    n = 200
    x = torch.rand(n, d)
    y = 4 * torch.sin(np.pi * x) * torch.cos(6 * np.pi * x**2)
    
    plt.scatter(x.numpy(), y.numpy())
    plt.title('plot of $f(x)$')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()
    ~~~

    ![img](data:image/png;base64,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)

    2.用pytorch搭建神经网络拟合数据集的曲线

    ~~~python
    # 搭建一个含有两层hidden layer的简单神经网络，激活函数为tanh
    
    model = nn.Sequential(
        nn.Linear(d, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )
    
    mse_loss = nn.MSELoss()
    
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    
    epochs = 10000
    for i in range(epochs):
        y_hat = model(x)
        loss = mse_loss(y_hat, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if(i % (epochs//10) == 0):
            print('iter={},\tloss={:.2f}'.format(i, loss))
            
    iter=0,	loss=3.40
    iter=1000,	loss=1.89
    iter=2000,	loss=1.36
    iter=3000,	loss=0.81
    iter=4000,	loss=0.36
    iter=5000,	loss=0.23
    iter=6000,	loss=0.15
    iter=7000,	loss=0.08
    iter=8000,	loss=0.03
    iter=9000,	loss=0.02
    
    # 注意到，如果对SGD加了Momentum，则收敛会快很多
    optim = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    iter=0,	loss=3.43
    iter=1000,	loss=0.02
    iter=2000,	loss=0.00
    iter=3000,	loss=0.00
    iter=4000,	loss=0.00
    iter=5000,	loss=0.00
    iter=6000,	loss=0.00
    iter=7000,	loss=0.00
    iter=8000,	loss=0.00
    iter=9000,	loss=0.00
    ~~~

    3.画出拟合的曲线，并与原始数据集的点对比，吻合地比较好

    ~~~python
    # 画数据集里的点
    plt.scatter(x.numpy(), y.numpy())
    plt.title('plot of $f(x)$ and $\hat{f}$(x)')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    # 画训练出来的函数曲线
    x_grid = torch.from_numpy(np.linspace(0, 1, 50)).float().view(-1, d)
    y_grid = model(x_grid)
    plt.plot(x_grid.detach().numpy(), y_grid.detach().numpy(), 'r')
    plt.show()
    ~~~

    ![img](data:image/png;base64,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)

- pytorch不会像Keras一样自动切validation set，需要自己切