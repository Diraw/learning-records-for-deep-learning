import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #选择GPU还是CPU

    data_transform = transforms.Compose( #构造预处理函数
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "../test.jpg" #加载测试图片
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img) #简单展示一下测试图片
    # [N, C, H, W]
    img = data_transform(img) #对图片进行预处理
    # expand batch dimension 扩充维度
    img = torch.unsqueeze(img, dim=0) #在图片数据最前面加一个batch维度

    # read class_indict
    json_path = "./class_indices.json"  # 读取一下json文件里面的分类信息到class_indict里边去
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=5).to(device) #这里的num_classes参数，在自己训练其他数据集的时候要改一下

    # load model weights
    weights_path = "./AlexNet.pth" #加载一下模型权重
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval() #关闭dropout
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu() #算一下输出
        predict = torch.softmax(output, dim=0) #算一下预测值
        predict_cla = torch.argmax(predict).numpy() #把预测值最大的索引拿出来，因为在下面我们想通过plt输出一下

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], #class类别 prob概率
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)): #在终端打印所有的类别和对应的概率
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
