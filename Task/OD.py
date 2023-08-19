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


labels = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

# 设置阈值和IOU（非极大抑制）阈值
conf_thres = 0.4
iou_thres = 0.5

# 读取图片
files_path = "dataset/COCO2017/val/images/" # 还没导入
image_files = []

for file in os.listdir(files_path):
    path = os.path.join (files_path, file)
    image = cv2.imread(path, file)
    #处理为640*640
    height, width, channels = image.shape
    tp = 640 / max(height, width)
    newH, newW = int(height * tp), int(width* tp)
    resized_image = cv2.resize(image, (newW, newH))

    # 转换
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).to(device)

    # 用模型检测
    pred = model(image)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5, classes=None, agnostic=False, max_det=100)  # 去除重叠框

    # 处理结果
    for index, value in enumerate (pred):
        if len(index) is not None and value is not None:
            for x1, y1, x2, y2, conf, cls in reversed (value):
                # 还原尺寸
                x1 = int(x1 * tp)
                y1 = int(y1 * tp)
                x2 = int(x2 * tp)
                y2 = int(y2 * tp)

                # 获取 置信度，






