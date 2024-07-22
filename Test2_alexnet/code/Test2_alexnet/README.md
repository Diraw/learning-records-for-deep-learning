> AlexNet.pth这个文件有55M，GitHub最多上传50M的文件，所以这个权重文件损坏了，懒得再pull下来删除了😂

> 主要是应该没人会来看我的repo吧hhh
# AlexNet
本文为本人学习深度学习所做，时间24/07/19

看的视频是b站up主霹雳吧啦Wz [2.1 pytorch官方demo(Lenet)](https://www.bilibili.com/video/BV1W7411T7qc/?spm_id_from=333.788&vd_source=0ac3c820aa67ba88616bd91e7b19b3d6)

代码也是他提供的 [他的GitHub链接](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

越学越觉得这个up讲的真挺好的，除了知识讲的很清晰，在视频里会随着系列的深入，逐渐加上一些项目工程的东西进去，像这个项目里就有断言、文件夹构成、跑一次模型的时间这些有用的小技巧的加入，真的讲的很好，十分收益
# 一、模型的代码
```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  
            # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  
            # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           
            # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  
            # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          
            # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          
            # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          
            # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  
            # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
```
`nn.Sequential()`可以将所有模块打包，这样会很方便

`self.features` 特征提取

`self.classifier`分类器

`nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)`深度3，大小48，卷积核大小11，步距4，padding 2，前两项是位置传参，后三项是关键字传参，这样代码更清晰一点（对我这种初学者来说更好理解hhh）
```python
class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int, #输入的深度
        out_channels: int, #使用卷积核的个数
        kernel_size: _size_2_t, #卷积核的大小
        stride: _size_2_t = 1, #步长、步距
        padding: Union[str, _size_2_t] = 0, #padding 四周补0处理
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True, #bias偏置，默认使用
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
```
padding这里是直接写成的2，视频里有，和论文不太准确，但方便，因为pytorch会自动处理，这就够了

`nn.ReLU(inplace=True)`这个inplace为True的话能够提高性能

`nn.Dropout(p=0.5)` dropout操作，p是随机失活的比例，默认为0.5

`nn.Linear(128 * 6 * 6, 2048)`
```python
class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
```
linear定义全连接层，可以看到第一个参数为输出神经元个数，第二个参数是输出神经元个数

`nn.Linear(2048, num_classes)`可以看到最终的输出个数是num_classes，这个是在最前面定义函数里
```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
```
也就是说我们定义模型的时候可以传参修改mun_classes，即最终的输出值，这个方便我们后续用这个模型去训练自己想训练的数据集

```python
if init_weights:
    self._initialize_weights()
```
定义模型的第二个参数init_weight 初始化权重，如果为True的话，进入下面代码
```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', 
                nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
```
其中用到了`self.modules()`，我们进入他的代码看看
```python
def modules(self) -> Iterator['Module']:
    r"""Return an iterator over all modules in the network."""
```
也就是说`self.modules()`会返回一个迭代器，这个迭代器会遍历网络中所有的模块，遍历我们在feature定义的所有层结构
```python
if isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(
        m.weight, mode='fan_out', 
        nonlinearity='relu')
```
如果该层是`nn.Conv2d`的话，就用`nn.init.kaiming_normal_`这个初始化方法去初始化m中的权重`m.weight`
```python
if m.bias is not None:
    nn.init.constant_(m.bias, 0)
```
如果它的偏置不为空的话，就用0进行初始化
```python
nn.init.normal_(m.weight, 0, 0.01)
```
正态分布初始化，均值为0，方差0.01

但其实我们不用去管这个`init_weights=False`，至少在当前版本pytorch会自动去初始化权重
```python
def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, start_dim=1)
    x = self.classifier(x)
    return x
```
forward定义正向传播过程

`x = torch.flatten(x, start_dim=1)`对图像进行展平

start_dim=1表示从0维之后展平，因为tensor的第一个维度是batch，我们把后面的深度高度宽度展开

也可以和lenet网络的代码一样，用view进行展平

# 二、训练的代码
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
```
选择使用GPU还是CPU，然后打印信息
```python
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  
                               # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
```
定义预处理函数

`transforms.RandomResizedCrop(224)`随机裁剪为 $224\times 224$ 的大小

`transform.RandomHorizontalFlip()`反转操作进行数据增强

`transforms.Resize((224, 224))`对训练集是随机裁剪，对验证集是resize，并且这里要输入两个变量

`os.getcwd()`获得当前所在目录

`os.path.join(os.getcwd(), "../..")`将当前目录和那个字符串连在一起，两个`..`代表返回上上级目录，这里的做法有利于项目工程

`data_root = os.path.abspath(os.path.join(os.getcwd(), "../..")) `
data根目录

`image_path = os.path.join(data_root, "data_set", "flower_data")`
获得图片的位置，后面也可以是`"/data_set/flower_data/"`

`os.path.exists(image_path)`:检查 image_path 是否存在

```python
train_dataset = datasets.ImageFolder(
                    root=os.path.join(image_path, "train"),
                    transform=data_transform["train"])
```
加载训练集的数据，可以看到里面的root和transform

`flower_list = train_dataset.class_to_idx` 加载每一个类别对应的索引值

`cla_dict = dict((val, key) for key, val in flower_list.items())`
将`类别 索引`变成`索引 类别`

```python
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)
```

`json.dumps()`: 这是 Python 的 json 模块中的一个函数，用于将 Python 对象（如字典、列表等）转换为 JSON 字符串

`cla_dict`: 这是一个 Python 字典对象，包含了你想要转换为 JSON 格式的数据

`indent=4`: 这是一个可选参数，用于指定 JSON 字符串的缩进级别。这里设置为 4，意味着生成的 JSON 字符串将使用 4 个空格进行缩进，从而使其更具可读性

```python
w = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  
# number of workers
print('Using {} dataloader workers every process'.format(nw))
```
`os.cpu_count()` 返回当前系统的 CPU 核心数量。例如，如果你的计算机有 8 个核心，这个函数将返回 8。

`atch_size if batch_size > 1 else 0` 这部分代码检查 `batch_size` 的值：如果 `batch_size` 大于 1，则返回 `batch_size`；如果 `batch_size` 小于或等于 1，则返回 0。

`8` 这是一个常数，表示工作线程的最大数量限制为 `8`

`min([...])`函数取上述三个值中的最小值，确定最终的工作线程数量
```python
train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=batch_size, shuffle=True,
                            num_workers=nw)

validate_dataset = datasets.ImageFolder(
                            root=os.path.join(image_path, "val"),
                            transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(
                            validate_dataset,
                            batch_size=4, shuffle=False,
                            num_workers=nw)
```
载入训练集，然后载入验证集，val_num验证集文件个数，训练集文件个数`train_num = len(train_dataset)`在前面
```python
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    test_data_iter = iter(validate_loader)
    test_image, test_label = next(test_data_iter)

    
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    imshow(utils.make_grid(test_image))
```
读取图片那的代码用.next方法现在好像会报错了`AttributeError: '_MultiProcessingDataLoaderIter' object has no attribute 'next'`，改一下用next()函数就好了

```python
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)
```
选择GPU还是CPU，定义损失函数，选择优化器，没什么好说的

`net.train()` 启用dropout方法

`net.eval()` 关闭dropout方法

记录训练时间
```python
t1=time.perf_couter()
print(time.perf_counter()-t1)
```

# 三、预测的代码
`img = torch.unsqueeze(img, dim=0)` 扩充维度，在最前面添加batch维度

`output = torch.squeeze(model(img.to(device))).cpu()`
这里使用squeeze压缩掉了第一个维度

`model = AlexNet(num_classes=5).to(device)`
这里的num_classes参数，在自己训练其他数据集的时候要改一下

`predict_cla = torch.argmax(predict).numpy()`
使用argmax获取概率最大的索引值

其余代码都比较简单，不说了，把预测的代码放下面，写了点注释：
```python
def main():
    device = torch.device("cuda:0" 
    if torch.cuda.is_available() else "cpu") #选择GPU还是CPU

    data_transform = transforms.Compose( #构造预处理函数
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "../test.jpg" #加载测试图片
    assert os.path.exists(img_path), "file: '{}' dose not exist."
    .format(img_path)
    img = Image.open(img_path)

    plt.imshow(img) #简单展示一下测试图片
    # [N, C, H, W]
    img = data_transform(img) #对图片进行预处理
    # expand batch dimension 扩充维度
    img = torch.unsqueeze(img, dim=0) #在图片数据最前面加一个batch维度

    # read class_indict
    json_path = "./class_indices.json"  
    # 读取一下json文件里面的分类信息到class_indict里边去
    assert os.path.exists(json_path), "file: '{}' dose not exist."
    .format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=5).to(device) 
    #这里的num_classes参数，在自己训练其他数据集的时候要改一下

    # load model weights
    weights_path = "./AlexNet.pth" #加载一下模型权重
    assert os.path.exists(weights_path), "file: '{}' dose not exist
    .".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval() #关闭dropout
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu() #算一下输出
        predict = torch.softmax(output, dim=0) #算一下预测值
        predict_cla = torch.argmax(predict).numpy() 
        #把预测值最大的索引拿出来，因为在下面我们想通过plt输出一下

    print_res = "class: {}   prob: {:.3}"
                .format(class_indict[str(predict_cla)], #class类别 prob概率
                        predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)): #在终端打印所有的类别和对应的概率
        print("class: {:10}   prob: {:.3}"
               .format(class_indict[str(i)],
                        predict[i].numpy()))
    plt.show()

```
