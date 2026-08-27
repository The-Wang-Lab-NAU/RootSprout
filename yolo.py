import colorsys
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, DecodeBoxNP,nms

import sys
import os

def get_base_path():
    """获取可执行文件或脚本所在目录（兼容 PyInstaller 打包）"""
    if getattr(sys, 'frozen', False):
        # 打包后，sys._MEIPASS 指向临时解压目录
        return sys._MEIPASS
    else:
        # 开发环境
        return os.path.dirname(os.path.abspath(__file__))


# class YOLO(object):
#     _defaults = {
       
#         "model_path"        : "model_data/Weights_yolo.pth",
#         "classes_path"      : 'model_data/voc_classes.txt',
#         "anchors_path"      : 'model_data/yolo_anchors.txt',
#         "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
#         "input_shape"       : [640, 640],
#         "phi"               : 'l',
#         "confidence"        : 0.5,
#         "nms_iou"           : 0.3,
#         "letterbox_image"   : True,
#         "cuda"              : True,
#     }
    
class YOLO(object):
    _defaults = {
       
        "model_path"        : os.path.join(get_base_path(), "model_data", "Weights_yolo.pth"),
        "classes_path"      : os.path.join(get_base_path(), "model_data", "voc_classes.txt"),
        "anchors_path"      : os.path.join(get_base_path(), "model_data", "yolo_anchors.txt"),

        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "input_shape"       : [640, 640],
        "phi"               : 'l',
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : True,
        "cuda"              : True,
    }
    
    
      
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

  
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)


        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self, onnx=False):

        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

   

    def detect_image(self, image, region_coord=True,seed_num="", crop = False, count = False):
      
        image_shape = np.array(np.shape(image)[0:2])
       
        image       = cvtColor(image)
       
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
      
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
           
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
           
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
        
            
                                           
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
          
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] - 45).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
      
    
        if count == True:
        
            print("top_label:", top_label)
            num=0
            for i in (top_label):
                if i ==1:
                    num+=1
            # print(num/len(top_label))
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            
            if num == 0:
                radio=0
            else:
                radio=num/len(top_label)
      
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
           
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                
                crop_image = image.crop([left, top, right, bottom])
                # crop_image.save(os.path.join(dir_save_path, file_name + "crop_" + str(i) + ".jpg"), quality=95, subsampling=0)
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".jpg"), quality=95, subsampling=0)
    #             print("save crop_" + str(i) + ".png to " + dir_save_path)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
       
        s=0
        seed_image_list=[]
        germinate_image_list =[]
       
        

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label = label.encode('utf-8')
            if predicted_class == "germinate":
             
                  crop_image = image.crop([left, top, right, bottom])
                  germinate_image_list.append(crop_image)
                  s=s+1
            if predicted_class == "not germinate":
                  crop_image = image.crop([left, top, right, bottom])
                  seed_image_list.append(crop_image)
                  s=s+1       

        Region_one = [] 
        Region_two = []
        Region_three =[]
        Region_four = []
        Region_five = []
        Region_six = []
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            #
            # 计算质心点坐标
            centroid=[np.int32((left+(right-left)/2)),np.int32((bottom-top)/2+top)]
            
            if region_coord == True:
          
          
                if region_coord[0][0] <centroid[0] < region_coord[0][2] and region_coord[0][1] < centroid[1] <region_coord[0][3] :
                    if predicted_class == "germinate":
                        Region_one.append(1)   
      
                elif region_coord[1][0] <centroid[0] < region_coord[1][2] and region_coord[1][1] < centroid[1] <region_coord[1][3] :
                    if predicted_class == "germinate":
                        Region_two.append(1)
           
                elif region_coord[2][0] <centroid[0] < region_coord[2][2] and region_coord[2][1] < centroid[1] <region_coord[2][3] :
                    if predicted_class == "germinate":
                        Region_three.append(1)
     
                elif region_coord[3][0] <centroid[0] < region_coord[3][2] and region_coord[3][1] < centroid[1] <region_coord[3][3] :
                    if predicted_class == "germinate":
                        Region_four.append(1)

                elif region_coord[4][0] <centroid[0] < region_coord[4][2] and region_coord[4][1] < centroid[1] <region_coord[4][3] :
                    if predicted_class == "germinate":
                        Region_five.append(1)
  
                elif region_coord[5][0] <centroid[0] < region_coord[5][2] and region_coord[5][1] < centroid[1] <region_coord[5][3] :
                    if predicted_class == "germinate":
                        Region_six.append(1)
                  

            if predicted_class == "not germinate":
                  for i in range(thickness):
                    # draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(0, 255, 0))
               
            elif predicted_class == "germinate":
                 
                 for i in range(thickness):
                  
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=(0, 0, 255))
                  
            del draw
        
        radio=[np.sum(Region_one)/int(seed_num),np.sum(Region_two)/int(seed_num),np.sum(Region_three)/int(seed_num),np.sum(Region_four)/int(seed_num),np.sum(Region_five)/int(seed_num),np.sum(Region_six)/int(seed_num)]
   
        return image,top_boxes,radio
    



