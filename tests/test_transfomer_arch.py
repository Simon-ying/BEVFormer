from mmdet.registry import MODELS
from mmengine import Config
import torch
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes
import numpy as np
import copy
import sys
sys.path.append(".")
'''
build models
'''
cfg = Config.fromfile("tests/test_config.py")
print(cfg.default_hooks)
img_backbone = MODELS.build(cfg.img_backbone)
img_backbone.eval()

img_neck = MODELS.build(cfg.img_neck)
img_neck.eval()

transformer = MODELS.build(cfg.transformer)
transformer.eval()

pts_bbox_head = MODELS.build(cfg.pts_bbox_head)
pts_bbox_head.eval()

bevformer = MODELS.build(cfg.model)
bevformer.eval()

'''
define parameters and fake input
'''
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

'''
bevformer preprocess
'''
imgs = batch_inputs_dict.get("imgs", None)
batch_input_metas = [item for item in batch_data_samples]
len_queue = imgs.size(1)
prev_img = imgs[:, :-1, ...]
imgs = imgs[:, -1, ...]
prev_img_metas = copy.deepcopy(batch_input_metas)

'''
image backbone and image neck
Output:
    img_feats, img_metas
'''
prev_bev = None
bs, len_queue, num_cams, C, H, W = prev_img.shape
imgs_queue = prev_img.view(bs*len_queue, num_cams, C, H, W)
img_feats_list = bevformer.extract_feat(
    imgs=imgs_queue,
    batch_input_metas=prev_img_metas,
    len_queue=len_queue)
img_metas = [each[f"{0}"] for each in prev_img_metas]
img_feats = [each_scale[:, 0] for each_scale in img_feats_list]

'''
bevhead preprocess
Input:
    img_feats, bev_queries,
    bev_h, bev_w, grid_length,
    bev_pos,
    img_metas,
    prev_bev
'''
bs, num_cam, _, _, _ = img_feats[0].shape
dtype = img_feats[0].dtype
object_query_embeds = pts_bbox_head.query_embedding.weight.to(dtype)
bev_queries = pts_bbox_head.bev_embedding.weight.to(dtype)

bev_mask = torch.zeros((bs, pts_bbox_head.bev_h, pts_bbox_head.bev_w),
                        device=bev_queries.device).to(dtype)
bev_pos = pts_bbox_head.positional_encoding(bev_mask).to(dtype)
grid_length=(pts_bbox_head.real_h / pts_bbox_head.bev_h,
             pts_bbox_head.real_w / pts_bbox_head.bev_w)
bev_h, bev_w = pts_bbox_head.bev_h, pts_bbox_head.bev_w
'''
transformer get bev features, TODO
'''


prev_bev = transformer.get_bev_features(
    img_feats,
    bev_queries,
    pts_bbox_head.bev_h,
    pts_bbox_head.bev_w,
    grid_length=(pts_bbox_head.real_h / pts_bbox_head.bev_h,
                 pts_bbox_head.real_w / pts_bbox_head.bev_w),
    bev_pos=bev_pos,
    img_metas=img_metas,
    prev_bev=prev_bev,
)


input = dict(
    inputs=batch_inputs_dict, 
    data_samples=batch_data_samples, 
    mode="tensor"
)
print(prev_bev.shape)
# onnx_program = torch.onnx.export(bevformer, input, "bevformer.onnx")
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