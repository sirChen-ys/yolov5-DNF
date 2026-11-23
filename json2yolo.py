# -*- coding: utf-8 -*-
# ==============================================================================
# [已废弃 / DEPRECATED] 请勿用于新的标注流程
#
# 本脚本是 labelme JSON -> YOLO txt 的早期单文件实现，功能已被
# prepare_dataset.py 整合（而 prepare_dataset.py 本身也已废弃）。
#
# 当前标注流程：Label Studio 标注 -> E:\Work\LabelStudio\export-yolo.ps1 导出
# 类别顺序统一由 D:\Project\yolov5-DNF\data\dnf.yaml 的 names 决定。
#
# 仅当需要转换历史 labelme 数据时才可能用到本脚本。
# ==============================================================================


import json
import os

# name2id = {'person':0,'helmet':1,'Fire extinguisher':2,'Hook':3,'Gas cylinder':4}
name2id = {'hero': 0, 'small_map': 1, "monster": 2, 'money': 3, 'material': 4, 'door': 5, 'BOSS': 6, 'box': 7, 'options': 8}
               
def convert(img_size, box):
    dw = 1./(img_size[0])
    dh = 1./(img_size[1])
    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
 
def decode_json(json_floder_path,json_name):
 
    txt_name = 'E:\\Computer_vision\\object_DNF\\datasets\\guiqi\\yolo5_datasets\\labels\\' + json_name[0:-5] + '.txt'
    txt_file = open(txt_name, 'w')
 
    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='gb2312'))
 
    img_w = data['imageWidth']
    img_h = data['imageHeight']
 
    for i in data['shapes']:
        
        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
            print(txt_name)
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])
 
            bb = (x1,y1,x2,y2)
            bbox = convert((img_w,img_h),bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')
    
if __name__ == "__main__":
    
    json_floder_path = r'E:\Computer_vision\object_DNF\datasets\guiqi\yolo5_datasets\labels_json'
    json_names = os.listdir(json_floder_path)
    for json_name in json_names:
        decode_json(json_floder_path,json_name)
