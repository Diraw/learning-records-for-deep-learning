import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def main():
    transform = transforms.Compose( #transform预处理函数
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片（导入训练集）
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, #data文件夹会创建在该文件的上一级，欧懂了，因为终端在E:\machine learning learning>
                                             download=True, transform=transform) #所以是E:\machine learning learning的当前目录下，所以是该文件的上一级
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片（导入测试集）
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=10000,
                                             shuffle=False, num_workers=0)

    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter) #将导入的测试集转化为迭代器，通过next一批一批获取数据

    classes = ('plane', 'car', 'bird', 'cat', #分类的十个类型
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet() #实例化model里面定义的模型
    loss_function = nn.CrossEntropyLoss() #定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001) #选用Adam优化器，传入网络中所有的参数，lr是学习率

    for epoch in range(5): #epoch代表循环迭代多少次

        running_loss = 0.0 #累加损失
        for step, data in enumerate(train_loader, start=0): #遍历训练集样本
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data #将输入的数据分为图像和标签

            # zero the parameter gradients
            optimizer.zero_grad() #用多次小batch达到大batch的效果

            # forward + backward + optimize
            outputs = net(inputs) #正向传播得到输出
            loss = loss_function(outputs, labels) #计算损失
            loss.backward() #进行反向传播
            optimizer.step() #进行参数更新

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:    #每隔500步。打印一次信息
                with torch.no_grad(): #在with块内的操作，不进行计算梯度，因为现在是测试集范围
                    outputs = net(val_image)  #用测试集进行正向传播
                    predict_y = torch.max(outputs, dim=1)[1] #在输出中寻找最大值，即预测的类别
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
                    # 将预测值和真实值进行比较，所有结果求和，此时是tensor类型，用item得到具体数值

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' % 
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    # 训练到第几轮了，每一轮的多少步，平均训练误差，准确率

                    running_loss = 0.0

    print('Finished Training')

    save_path = 'Lenet240717.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
