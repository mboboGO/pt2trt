from easydict import EasyDict as edict

config = edict()

# data info
test_img_dir = '/home/bobmin/fbp/datasets/actor_ranking/facecrop256'
img_list = '/home/bobmin/fbp/datasets/actor_ranking/val_list_mbobo.json'

# model info
resize_size = [256,256]
crop_size = [224,224]
pth_path = '/home/bobmin/fbp/fbp_pt/output/run_mbn2_sig_adam_actor/fbp_mbn2_0.0517.model'

# pth2onnx
onnx_path = 'output/fbp_model.onnx'

# onnx2trt
trt_path = 'output/fbp_model.trt'

