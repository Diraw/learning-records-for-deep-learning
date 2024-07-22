# VGGNet
本文为本人学习深度学习所做，时间24/07/22

看的视频是b站up主霹雳吧啦Wz [4.2 使用pytorch搭建VGG网络](https://www.bilibili.com/video/BV1i7411T7ZN/?spm_id_from=333.788&vd_source=0ac3c820aa67ba88616bd91e7b19b3d6)

代码也是他提供的 [他的GitHub链接](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

VGG主要做一个过渡，尽快进入后面ResNet和EffectiveNet的学习

另外我的笔记和源码放在了我的GitHub仓库 [我的GitHub链接](https://github.com/Diraw/learning-records-for-deep-learning)
# 一、模型的脚本
```python
'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
```
数字为卷积核个数，M为池化层

`def make_features(cfg: list):`到时候直接传入对应配置的列表就可以了
```python
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)
```
`layers = []`定义空列表用来存放每一层的结构

`in_channels`最开始是3，后面就会变`in_channels = v`

`nn.ReLU(True)`True就是inplace为True

`return nn.Sequential(*layers)`将列表以非关键字参数传给nn.Sequential，因为每个模型的层数不一样嘛（nn.Sequential接受一个非关键字参数、或者一个有序的字典）

`make_features`这个函数定义用于提取特征的网络结构，而分类网络结构（那三个全连接层）`classifier`同这里分开定义
```python
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()
```
`features`传入刚才定义的提取特征层结构

`num_classes=1000`传入分类的数目

`init_weights=False`是否初始化权重

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
```
初始化权重函数

```python
def forward(self, x):
    # N x 3 x 224 x 224
    x = self.features(x)
    # N x 512 x 7 x 7
    x = torch.flatten(x, start_dim=1)
    # N x 512*7*7
    x = self.classifier(x)
    return x
```
正向传播过程

`x = self.features(x)`进行特征提取

`x = torch.flatten(x, start_dim=1)`进行展平操作，略过第0个维度batch

`x = self.classifier(x)`进行图像分类
```python
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
```
用于实例化vgg网络，默认是vgg16

`model = VGG(make_features(cfg), **kwargs)`**kwargs传入一个可变长度的字典，即设置`num_classes=1000, init_weights=False`这些参数

# 二、训练的脚本
```python
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
```
预处理函数

`RandomResizedCrop`随机裁剪

`RandomHorizontalFlip`随机水平翻转

`ToTensor`转化为tensor类型

`Normalize`标准化处理

```python
net = vgg(model_name=model_name, num_classes=5, init_weights=True)
```
实例化vgg，后面两个就是前面说的可变长度字典参数

其他的和AlexNet的都差不多，就不讲了，因为不做迁移学习，所以vgg只做一个过渡，尽快进入后面ResNet和EffectiveNet的学习