# pt2trt

## Requirements
py3.6
pytorch
tensorrt v7
onnx
onnxsim
pycuda

## onnx&prt install
# pycuda
pip install pycuda
# onnx
    ```shell
    pip install onnx
    pip install onnx-simplifier
    ```

# trt 
Notably, the trt version should be compatible with the py version:
1. Downloading tensorrt in 'https://developer.nvidia.com/nvidia-tensorrt-7x-download'
2. tar xzvf xxx & cd TensorRT-7.x.x.x/python & pip install tensorrt-7.x.x.x-cpxx-none-linux_x86_64.whl

## pth to onnx
1. convert pth to onnx:
    changing config file in config.py
    modifying the function of load_model pth2onnx/pth_to_onnx.py
    ``` shell
    python pth2onnx/pth_to_onnx.py
    python pth2onnx/inference_onnx.py
    ```

2. testing the onnx
    ``` shell
    python pth2onnx/inference_onnx.py
    ```

## pth to trt
1. convert onnx to trt:
    changing config file in config.py
    modifying the function of load_model onnx2trt/onnx2trt.py
    ``` shell
    python onnx2trt/onnx_to_trt.py
    python onnx2trt/inference_trt.py
    ```

2. testing the trt
    ``` shell
    python onnx2trt/inference_trt.py
    ```
