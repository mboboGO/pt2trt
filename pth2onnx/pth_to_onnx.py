# # coding: utf-8
from os import path as osp
import os
import config

import torch
import onnx
from onnxsim import simplify

from mbn2 import MobileNetV2

def pth_to_onnx(net, onnx_path, input_h, input_w):
    inputs = torch.rand(1, 3, input_h, input_w)
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(net, inputs, onnx_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=9)

def simplify_onnx(onnx_path):
    print(f'\nsimplify | {onnx_path}')
    model_simp, check_ok = simplify(onnx_path, check_n=5)
    if check_ok:
        print('successful!')
        onnx_simp_path = osp.splitext(onnx_path)[0] + '_simp.onnx'
        onnx.save(model_simp, onnx_simp_path)
        print(f'save | {onnx_simp_path}')
    else:
        print('failed!')

def load_model(pth_path):
    net = MobileNetV2()
    checkpoint = torch.load(pth_path)
    net.load_state_dict(checkpoint['state_dict'])
    return net    

if __name__ == '__main__':
    # load model
    net = load_model(config.pth_path)
    print('==> loading pth model finished!')
    # pth2onnx
    if not os.path.exists(config.onnx_path):
        pth_to_onnx(net,config.onnx_path,config.crop_size[0],config.crop_size[1])
    print('==> pth to onnx finished!')
    simplify_onnx(config.onnx_path)
    print('==> simplify onnx finished!')
