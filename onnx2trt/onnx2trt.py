import config
from os import path as osp
import os
import tensorrt as trt

import common

TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path, engine_file_path):
    """ create and save a TensorRT engine from an ONNX file """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(
            network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28  # 256MiB
        builder.max_batch_size = 1
        if not osp.exists(onnx_file_path):
            raise ValueError(f'error | cannot find ONNX file | {onnx_file_path}')

        print(f'\nparse ONNX file | {onnx_file_path}')
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(f'error | {parser.get_error(error)}')
                return None

        print('\nbuild TensorRT engine | this may take a while...')
        network.get_input(0).shape = [1, 3, config.crop_size[0], config.crop_size[1]]  # input size NCHW
        engine = builder.build_cuda_engine(network)

        print(f'\nsave TensorRT engine | {engine_file_path}')
        with open(engine_file_path, 'wb') as f:
            f.write(engine.serialize())
        return engine


if __name__ == '__main__':
    if not os.path.exists(config.trt_path):
        onnx_simp_path = config.onnx_path.replace('.onnx','_simp.onnx')
        engine = build_engine(onnx_simp_path,config.trt_path)


