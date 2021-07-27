# coding: utf-8
# created by tianyuqiu on 20210722
import os
from os import path as osp
import time
import numpy as np
import cv2
import json
import config

import onnxruntime as ort

# input size
resize_size = config.resize_size
crop_size = config.crop_size


def read_image(image_path):
    '''
     loading image:
        the data augmentation includes:
            Resize()
            CenterCrop()
            Normalize()
    '''
    image = cv2.imread(image_path)  # BGR, HWC
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Albumentations uses RGB format
    h, w = image.shape[:2]
    # 保持长宽比resize，不够的补黑
    scale = 1.0
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    if h != resize_size[0] or w != resize_size[1]:
        scale = min(resize_size[0] / h, resize_size[1] / w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h_resize, w_resize = image.shape[:2]
        if w_resize < resize_size[1]:
            pad_left = (resize_size[1] - w_resize) // 2
            pad_right = resize_size[1] - w_resize - pad_left
        if h_resize < resize_size[0]:
            pad_top = (resize_size[0] - h_resize) // 2
            pad_bottom = resize_size[0] - h_resize - pad_top
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, 0)
    # img crop
    x = int(image.shape[0]/2 - crop_size[0]/2)
    y = int(image.shape[1]/2 - crop_size[1]/2)
    crop_img = image[x:x+crop_size[0], y:y+crop_size[1],:]
    # norm
    crop_img = crop_img.astype(np.float32) / 255
    crop_img[:,:,0] = (crop_img[:,:,0]-0.485)/0.229;  
    crop_img[:,:,1] = (crop_img[:,:,1]-0.456)/0.224;
    crop_img[:,:,2] = (crop_img[:,:,2]-0.406)/0.225;

    return crop_img


def inference(onnx_file_path, image_path_list):
    ort_session = ort.InferenceSession(onnx_file_path)

    # ort_session第一次load到gpu会很慢，跑一次假的，方便之后真实图片计时
    inputs = np.random.rand(1, 3, crop_size[0], crop_size[1]).astype(np.float32)
    ort_outputs = ort_session.run(None, {'input': inputs})

    time_sum = 0
    time_count = 0
    for image_path in image_path_list:
        print('process image | {}'.format(image_path))
        t_start = time.time()
        image = read_image(image_path)
        inputs = image.transpose(2, 0, 1)[np.newaxis]  # HWC to CHW to NCHW
        ort_outputs = ort_session.run(None, {'input': inputs})
        pred = ort_outputs[0]
        print('output: {}'.format(pred))
        time_sum += time.time() - t_start
        time_count += 1
        print('cur_time={:.2f}ms total_time={:.2f}ms'.format((time.time() - t_start) * 1000,time_sum/time_count*1000))

def load_json():
    imlist = json.load(open(config.img_list,'r'))['labels']
    images = []
    for imname in imlist.keys():
        _imname = imname[7:]
        path = os.path.join(config.test_img_dir,_imname)
        images.append(path)
    return images

def load_txt():
    images = []
    fid = open(config.img_list,'r')
    for line in fid.readlines():
        data = line.strip('\n').split(' ')
        path = os.path.join(data_path,data[0])
        images.append(path)
    fid.close()
    return images


if __name__ == '__main__':
    # load test image
    if config.img_list.endswith(".json"):
        image_path_list = load_json()
    elif config.img_list.endswith(".txt"):
        image_path_list = load_json() 
    # inference
    onnx_simp_path = config.onnx_path.replace('./onnx','simp.onnx')
    inference(onnx_simp_path, image_path_list)
    print('==> testing finished!')
