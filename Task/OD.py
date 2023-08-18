import os

import torch
import cv2
import numpy as np

from models.objectDetection import non_max_suppression
from models.objectDetection import attempt_load





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

weights = 'models/objectDetection/yolov5s.pt'
model = attempt_load(weights)  # 读取yolo模型
model.to(device)

# 设置阈值和IOU（非极大抑制）阈值
conf_thres = 0.4
iou_thres = 0.5

# 读取图片
files_path = "dataset/COCO2017/val/images/" # 还没导入
image_files = []

for file in os.listdir(files_path):
    path = os.path.join (files_path, file)
    image = cv2.imread(path, file)
    # 转换
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).to(device)

    # 用模型检测
    pred = model(image)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False, max_det=100)  # 去除重叠框

    # 处理结果
