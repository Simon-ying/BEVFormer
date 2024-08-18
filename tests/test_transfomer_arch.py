from mmdet.registry import MODELS
from mmengine import Config
import torch
from torchviz import make_dot
import hiddenlayer as hl
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes
import numpy as np

cfg = Config.fromfile("test_config.py")
img_backbone = MODELS.build(cfg.img_backbone)
img_backbone.eval()
img_neck = MODELS.build(cfg.img_neck)
img_neck.eval()
transformer = MODELS.build(cfg.transformer)
transformer.eval()

bevformer = MODELS.build(cfg.model)
bevformer.eval()

bs = 1
num_views = 1
channel = 3
H, W = 128, 128
len_queue = 2

fake_imgs = torch.rand(bs, len_queue, num_views, channel, H, W)
batch_inputs_dict = dict(
    imgs=fake_imgs
)

fake_meta_info = {
    "0": dict(
        prev_bev_exists=False,
        can_bus=[0 for _ in range(18)],
        lidar2img=np.reshape(np.identity(4), (1,4,4)).tolist(),
        img_shape=[[H, W]]
    ),
    "1": dict(
        prev_bev_exists=True,
        can_bus=[0 for _ in range(18)],
        lidar2img=np.reshape(np.identity(4), (1,4,4)).tolist(),
        img_shape=[[H, W]]
    ),
}
fake_data_sample = Det3DDataSample(metainfo=fake_meta_info)
batch_data_samples = [fake_meta_info]
# output = bevformer(batch_inputs_dict, batch_data_samples, mode="tensor")
# print(batch_inputs_dict, batch_data_samples)
onnx_program = torch.onnx.export(bevformer, (batch_inputs_dict, batch_data_samples), "bevformer.onnx")
# for key, value in output.items():
#     try:
#         print(f"{key} has shape: {value.shape}")
#     except:
#         pass
# B, N, C, H, W = fake_imgs.size()
# fake_imgs = fake_imgs.view(B * N, C, H, W)

# img_feats = img_backbone(fake_imgs)

# img_feats = img_neck(img_feats)

# torch.onnx.export(img_neck, img_feats, "img_neck.onnx", opset_version=11)
# graph = hl.build_graph(img_neck, img_feats)
# graph.save("img_neck.png")
# with torch.no_grad():
#     img_feats = img_backbone(fake_imgs)
# dot = make_dot(img_feats, params=dict(img_backbone.named_parameters()))
# dot.render("img_backbone", format="png")
# 

# img_feats = img_neck(img_feats)


# print(transformer)