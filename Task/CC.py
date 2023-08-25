import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


def get_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

# 加载预训练的ResNet50模型
model = models.resnet50(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载并预处理待分类的图像
# 读取图片
files_path = "dataset/COCO2017/val/images/" # 还没导入

target_list = []

# 标签
labels={
    1,
    2,
    3,
}

TP = []
FP = []
for file in os.listdir(files_path):

    # 读标签
    name, extension = os.path.splitext(file)
    label_file_name = name + '.txt'
    #标签
    gts = get_file_content(label_file_name)


    # 加个张量
    file = file.unsqueeze(0)
    # 将模型移动到设备
    model = model.to(device)
    file = file.to(device)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(file)

    # 获取预测结果
    _, predicted_idx = torch.max(outputs, 1)

    # 打印预测结果
    predicted_number = predicted_idx.item()
    if gts == predicted_number:
        TP.append(1)
        FP.append(0)
    else:
        FP.append(1)
        TP.append(0)

recall = TP/(TP+FP)


