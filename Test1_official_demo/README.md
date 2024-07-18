# Lenet
本文为本人学习深度学习所做，时间24/07/18

看的视频是b站up主霹雳吧啦Wz [2.1 pytorch官方demo(Lenet)](https://www.bilibili.com/video/BV187411T7Ye/?spm_id_from=333.788&vd_source=0ac3c820aa67ba88616bd91e7b19b3d6)

代码也是他提供的 [他的GitHub链接](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

# 一、模型的代码
首先是看`model.py`
```python
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):

    def __init__(self): 
        super(LeNet, self).__init__() # 调用父类（基类）的构造函数
        self.conv1 = nn.Conv2d(3, 16, 5) #定义卷积层
        self.pool1 = nn.MaxPool2d(2, 2) #定义池化层层
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x
```
## 1、第一个卷积层
```python
class Conv2d(_ConvNd):
    __doc__ = r"""Applies a 2D convolution 
    over an input signal composed of several input planes."""
```


$$
\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
$$


```python
def __init__(
        self,
        in_channels: int, #第一个参数：输入的深度
        out_channels: int, #第二个参数：使用卷积核的个数
        kernel_size: _size_2_t, #第三个参数：卷积核的大小
        stride: _size_2_t = 1, #步长、步距默认为1
        padding: Union[str, _size_2_t] = 0, #padding 四周补0处理
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True, #bias偏置，默认使用
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    )
```
pytorch tenser的通道顺序：[batch,channel,height,width]

`self.conv1 = nn.Conv2d(3, 16, 5)`

第一个参数：输入的深度3，因为输入的是彩色图片，RGB有三个channel

第二个参数：卷积核的个数是16

第三个参数：卷积核的尺寸是 $5\times 5$

根据公式：经卷积后的矩阵尺寸大小：$N=(W-F+2P)/S+1$

> 1.输入图片大小 $W\times W$
> 2.Filter 大小 $F\times F$
> 3.步长 $S$
> 4.padding 的像素数 $P$

`x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)`

input(3, 32, 32)，输入是一个 $3\times 32\times 32$ 的图片，所以 $W=32$

卷积核的大小 $F=5$

padding $P=0$

stride $S=1$

带进去算的话，$N=(32-5+0)/1+1=28$

所以输出矩阵尺寸是 $28\times 28$ 的，output(16, 28, 28)

因为我们使用了16个卷积核，所以第一个参数channel变成了16

pytorch tenser的通道顺序：[batch,channel,height,width]，这里是没有把batch写出来的，只写了channel、高度、宽度

## 2、第二个池化层
`self.pool1 = nn.MaxPool2d(2, 2)`
`class MaxPool2d(_MaxPoolNd):`没有定义构造函数，遂进入父类

```python
class _MaxPoolNd(Module):

    def __init__(self, 
        kernel_size: _size_any_t,  #第一个参数：池化核的大小
        stride: Optional[_size_any_t] = None, 
        #第二个参数：步距，默认为None，看下面
        padding: _size_any_t = 0,  #padding默认为0
        dilation: _size_any_t = 1,
        return_indices: bool = False, 
        ceil_mode: bool = False) -> None:

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        #如果不指定步距的话，就会默认有kernel_size作为步距
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
```
`self.pool1 = nn.MaxPool2d(2, 2)`

第一个参数：池化核大小 $2\times 2$

第二个参数：步距为2

在上一个卷积层我们的输出为`output(16, 28, 28)`

经过池化层之后`x = self.pool1(x) # output(16, 14, 14)`

池化层不改变矩阵的深度，只改变高度和宽度，所以深度依然是16，高和宽减半

## 3、第二个卷积层
`self.conv2 = nn.Conv2d(16, 32, 5)`

因为上一个池化层出来的深度为16，所以第一个参数channel变成了16

我们采用32个卷积核，每个卷积核尺寸为5

根据公式：经卷积后的矩阵尺寸大小：$N=(W-F+2P)/S+1$

> 1.输入图片大小 $W\times W$
> 2.Filter 大小 $F\times F$
> 3.步长 $S$
> 4.padding 的像素数 $P$

输出图片的大小为 $14\times 14$

padding看池化层的代码，默认为0，步距看卷积层的代码，默认为1

因此输出的矩阵尺寸为(14-5+0)/1+1=10

所以输出`x = F.relu(self.conv2(x)) # output(32, 10, 10)`

可以看到输出`output(32, 10, 10)`是 $32\times 10\times 10$

## 4、第二个池化层
`self.pool2 = nn.MaxPool2d(2, 2)`

池化核个数为2，步距为2

`x = self.pool2(x) # output(32, 5, 5)`

输出深度不变，尺寸减半

## 5、三个全连接层

全连接层的输入是一个一维的向量，所以我们需要把我们的 $32\times 5\times 5$ 的特征矩阵展成1维

所以第一个全连接层`self.fc1 = nn.Linear(32*5*5, 120)`，输出的节点是 $32\times 5\times 5$ 这么多个，然后他自己的节点个数我们设置为是120

后面两行代码便不难理解

```python
self.fc2 = nn.Linear(120, 84)
self.fc3 = nn.Linear(84, 10)
```
最后的输出为10，是因为我们使用的是CIFAR-10这个数据集，有10个类别

## 6、正向传播过程
```python
def forward(self, x):
    x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
    x = self.pool1(x)            # output(16, 14, 14)
    x = F.relu(self.conv2(x))    # output(32, 10, 10)
    x = self.pool2(x)            # output(32, 5, 5)
    x = x.view(-1, 32*5*5)       # output(32*5*5)
    x = F.relu(self.fc1(x))      # output(120)
    x = F.relu(self.fc2(x))      # output(84)
    x = self.fc3(x)              # output(10)
    return x
```
`F.relu()`是RELU激活函数

我们通过`x = x.view(-1, 32*5*5)`这个view方法将矩阵展成一维，由于x的形状是`(batch_size, 32, 5, 5)`，因此总元素数量是`batch_size * 32 * 5 * 5`，-1告诉PyTorch自动推断出第一个维度（batch）的大小，以确保张量的总元素数量保持不变

这里我们没有用softmax函数，是因为我们这个模型使用的loss函数里自带了一个softmax函数

# 二、训练代码
## 1、导入训练集
```python
transform = transforms.Compose( #transform预处理函数
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```
先定义了一个预处理函数

`Compose`将处理打包成一个整体

```python
class ToTensor:
    """Convert a PIL Image or ndarray to tensor 
    and scale the values accordingly."""
```
`ToTensor`将图像数据转化为tensor数据
```
Converts a PIL Image or numpy.
ndarray (H x W x C) in the range[0, 255] to a torch.
FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
```

一个是变换维度的顺序，一个是进行归一化

```python
class Normalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.

    output[channel] = (input[channel] - mean[channel]) / std[channel]"""

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        _log_api_usage_once(self)
        self.mean = mean
        self.std = std
        self.inplace = inplace
```

`Normalize`进行标准化，mean为均值，std为标准差

```python
    # 50000张训练图片（导入训练集）
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform)
```

`root`为下载数据的保存地址，这里的相对路径指的是你的终端所在位置的相对路径，并不是你的文件所在位置的相对路径，如果你用vscode打开的外层文件夹，他会把这个文件下载到外面去

`train`如果为True，导入的就是训练集

`download`其实可以一直为true，它会自动检验

`transform`使用我们刚才定义的预处理函数

```python
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=36,
        shuffle=True, num_workers=0)
```

将训练集导入分为一个批次一个批次的数据

`batch_size`每一批次36张图片

`shuffle`是否要将数据集打乱，一般为True

`num_workers`在windows环境下必须为0

## 2、导入测试集
```python
    val_set = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=False, transform=transform)
```

`train`为False，所以导入的是测试集

```python
val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=10000,
        shuffle=False, num_workers=0)
```

`batch_size`设置为10000，一次性全部导入

```python
val_data_iter = iter(val_loader)
val_image, val_label = next(val_data_iter)
```

将导入的测试集转化为迭代器，通过next一批一批获取数据

## 3、其他准备
```python
classes = ('plane', 'car', 'bird', 'cat', 
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

导入分类的十个类型，index0对应的plane

```python
net = LeNet() #实例化model里面定义的模型
loss_function = nn.CrossEntropyLoss() #定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001) 
#选用Adam优化器，传入网络中所有的参数，lr是学习率learning rate
```

```python
class BCEWithLogitsLoss(_Loss):
    r"""This loss combines a `Sigmoid` layer and 
        the `BCELoss` in one singleclass.
    Note that this case is equivalent to applying 
        :class:`~torch.nn.LogSoftmax`on an input, 
        followed by :class:`~torch.nn.NLLLoss`."""
```

这个损失函数里包含了softmx函数的效果

## 4、训练过程
```python
for epoch in range(5): #epoch代表循环迭代多少次
    running_loss = 0.0 #累加损失
    for step, data in enumerate(train_loader, start=0): 
    #遍历训练集样本
        inputs, labels = data #将输入的数据分为图像和标签
        optimizer.zero_grad() #用多次小batch达到大batch的效果

        outputs = net(inputs) #正向传播得到输出
        loss = loss_function(outputs, labels) #计算损失
        loss.backward() #进行反向传播
        optimizer.step() #进行参数更新

        running_loss += loss.item()

        if step % 500 == 499:    #每隔500步。打印一次信息
            with torch.no_grad(): 
            #在with块内的操作，不进行计算梯度，因为现在是测试集范围
                outputs = net(val_image)  #用测试集进行正向传播
                predict_y = torch.max(outputs, dim=1)[1] 
                #在输出中寻找最大值，即预测的类别
                accuracy = torch.eq(predict_y, val_label)
                .sum().item() / val_label.size(0)
                # 将预测值和真实值进行比较，所有结果求和
                # 此时是tensor类型，用item得到具体数值

                print('[%d, %5d] train_loss: %.3f 
                      test_accuracy: %.3f' % 
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                # 训练到第几轮了，每一轮的多少步，平均训练误差，准确率

                running_loss = 0.0
```
# 3、预测代码
```python
def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), 
        #我们的网络输入必须是32*32的，所以必须reshape缩放一下
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet() #实例网络
    net.load_state_dict(torch.load('Lenet.pth')) 
    #载入训练好的权重文件

    im = Image.open('test.jpg') #载入图像
    im = transform(im)  
    # [C, H, W] 进行预处理，得到[channel 高度 宽度]的格式
    im = torch.unsqueeze(im, dim=0)  
    # [N, C, H, W] 在最前面(dim=0)增加一个维度
    # 因为tenser的通道顺序：[batch,channel,height,width]

    with torch.no_grad(): #使用with不计算损失梯度
        outputs = net(im) #将图像传入网络得到输出
        predict = torch.max(outputs, dim=1)[1].numpy() #得到预测值
    print(classes[int(predict)]) #预测值对应classes为识别结果
```
