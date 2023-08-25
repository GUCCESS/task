import os

import torch
import cv2
import numpy as np

from models.objectDetection import non_max_suppression
from models.objectDetection import attempt_load


def get_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content
def calculate_iou(box1, box22):
    #计算box1的坐标
    x1, y1, w1, h1 = box1
    lb, x2, y2, w2, h2 = box22

    # 计算box1和box2的右下角坐标
    x1_right, y1_bottom = x1 + w1, y1 + h1
    x2_right, y2_bottom = x2 + w2, y2 + h2

    # 计算交集的坐标
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_intersection_right = min(x1_right, x2_right)
    y_intersection_bottom = min(y1_bottom, y2_bottom)

    # 计算交集的面积
    intersection_area = max(0, x_intersection_right - x_intersection) * max(0, y_intersection_bottom - y_intersection)

    # 计算box1和box2的面积
    box1_area = w1 * h1
    box2_area = w2 * h2

    # 计算并返回IOU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return lb, iou

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

target_list = []
for file in os.listdir(files_path):

    # 获取文件对应的标签
    name, extension = os.path.splitext(file)
    label_file_name = name + '.txt'
    label_file = get_file_content(label_file_name)
    # 标签边界框
    gts = label_file.splitlines()

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

                isused = [0] * len(gts)
                tp = 0
                before_iou = 0
                cls = 0
                # 和对应图片标签不断对比 判断tp还是fp  x1x2y1y2跟标签中的x1x2y1y2进行对比 （怎么比较）
                for i, gt in enumerate(gts):
                    lb, iou = calculate_iou([x1,y1,x2,y2], gt)
                    if iou > before_iou:
                        before_iou =iou
                        cls = lb
                        tp = i
                # iou>0.5 且没被检测过 标记为tp（1）
                if iou >= iou_thres and isused[tp] == 0 :
                    sign = 1
                else:
                    sign =0
                # 添加 tp/fp，置信度，分类
                target_list.append(sign, conf, cls)
                isused[tp] = 1


# 计算p r
# 按照confidence排序
sorted_target_list = sorted(target_list, key = lambda x:x[1], reverse=True)

tp_, conf_, cls_ = [row[0] for row in sorted_target_list], [row[1] for row in sorted_target_list], [row[2] for row in sorted_target_list]
TP = []
FP = []
for i in tp_:
    if i == 1:
        TP.append(1)
        FP.append(0)
    else:
        TP.append(0)
        FP.append(1)
acc_TP = np.cumsum(TP)
acc_FP = np.cumsum(FP)
# p r
precision = acc_TP / (acc_TP + acc_FP)
recall = acc_TP / len(acc_TP)

# ap
mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
mpre = np.concatenate(([1.], precision, [0.]))

# pr曲线 插值算ap
x = np.linspace(0, 1, 101)
ap = np.trapz(np.interp(x, mrec, mpre), x)

map = sum(ap) / len(acc_TP)














