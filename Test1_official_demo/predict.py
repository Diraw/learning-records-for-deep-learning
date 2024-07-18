import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), #我们的网络输入必须是32*32的，所以必须reshape缩放一下
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet() #实例网络
    net.load_state_dict(torch.load('Lenet.pth')) #载入训练好的权重文件

    im = Image.open('test.jpg') #载入图像
    im = transform(im)  # [C, H, W] 进行预处理，得到[channel 高度 宽度]的格式
    im = torch.unsqueeze(im, dim=0)  
    # [N, C, H, W] #在最前面(0)增加一个维度，因为pytorch tenser的通道顺序：[batch,channel,height,width]

    with torch.no_grad(): #使用with不计算损失梯度
        outputs = net(im) #将图像传入网络得到输出
        predict = torch.max(outputs, dim=1)[1].numpy() #得到预测值
    print(classes[int(predict)]) #预测值对应classes为识别结果


if __name__ == '__main__':
    main()
