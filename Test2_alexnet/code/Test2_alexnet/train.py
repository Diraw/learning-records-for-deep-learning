import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #选择GPU和CPU
    print("using {} device.".format(device))

    data_transform = { #定义预处理函数，用个字典更方便
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  #获得data的根目录
    image_path = os.path.join(data_root, "data_set", "flower_data")  #获得数据集的目录
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), #载入训练集
                                         transform=data_transform["train"])
    train_num = len(train_dataset) #获得训练集文件个数

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx #获得类别，如上
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4) # 将类别的信息写入json文件中
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers 核心数目
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, #载入训练集
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), #载入验证集
                                            transform=data_transform["val"])
    val_num = len(validate_dataset) #获得验证集文件个数
    validate_loader = torch.utils.data.DataLoader(validate_dataset, #载入验证集
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # # test_data_iter = iter(validate_loader)
    # # test_image, test_label = test_data_iter.next()
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = next(test_data_iter)

    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True) #实例化网络

    net.to(device) #选择GPU或者CPU
    loss_function = nn.CrossEntropyLoss() #加载损失函数
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002) #选择adam优化器

    epochs = 10 #循环10次
    save_path = './AlexNet.pth' #模型保存路径
    best_acc = 0.0 #历史最佳正确率
    train_steps = len(train_loader) #获得每一次循环的训练总步数
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout) # 创建一个进度条，用于显示训练过程中批次（batch）的处理进度
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()  # 用多次小batch达到大batch的效果
            outputs = net(images.to(device))  # to(device)指定到GPU或者CPU上，进行正向传播
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()  # 进行参数更新

            # print statistics
            running_loss += loss.item()

            # train_bar.desc 是 tqdm 进度条的描述文本属性
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, 
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval() #关闭dropout
        acc = 0.0  # accumulate accurate number / epoch 用来计算预测正确的次数
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout) #创建进度条
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc: #如果这一次正确率变高了，就保存模型
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
