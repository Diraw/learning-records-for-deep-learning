# os:提供与操作系统交互的功能，如文件和目录操作
import os 
# shutil:提供高级的文件操作功能，如复制和删除文件
from shutil import copy, rmtree 
# random: 提供生成随机数和随机选择的功能
import random

# mk_file 函数用于创建一个新的文件夹
def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 定义验证集所占总数据集的比例，这里设置为10%
    split_rate = 0.1

    # 指向你解压后的flower_photos文件夹
    # cwd获得当前工作目录
    cwd = os.getcwd() 
    # data_root数据根目录，假设在当前工作目录下的flower_data文件夹中
    data_root = os.path.join(cwd, "flower_data")
    # origin_flower_path原始花卉图片的路径
    origin_flower_path = os.path.join(data_root, "flower_photos")
    # assert确保原始花卉图片路径origin_flower_path存在，否则抛出错误
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    # flower_class获取原始数据集中所有类别的名称（每个类别是一个文件夹）
    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    for cla in flower_class:
        # 获取类别文件夹的路径
        cla_path = os.path.join(origin_flower_path, cla)
        # 获取类别文件夹中的所有图片
        images = os.listdir(cla_path)
        # 计算图片数量
        num = len(images)
        # 随机选择10%的图片作为验证集
        eval_index = random.sample(images, k=int(num*split_rate))
        # 遍历所有图片，将其复制到训练集或验证集的相应目录中
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            # 打印处理进度
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        # 换行
        print()
    # 打印处理完成的信息
    print("processing done!")


if __name__ == '__main__':
    main()
