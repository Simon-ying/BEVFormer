from mmdet.registry import MODELS
from mmengine import Config
import torch
from torchviz import make_dot
import hiddenlayer as hl

cfg = Config.fromfile("test_config.py")
img_backbone = MODELS.build(cfg.img_backbone)
img_backbone.eval()
img_neck = MODELS.build(cfg.img_neck)
img_neck.eval()
transformer = MODELS.build(cfg.transformer)
transformer.eval()

bevformer = MODELS.build(cfg.model)
bevformer.eval()

bs = 2
num_views = 1
channel = 3
H, W = 256, 256
len_queue = 2

fake_imgs = torch.rand(bs, num_views, channel, H, W)
B, N, C, H, W = fake_imgs.size()
fake_imgs = fake_imgs.view(B * N, C, H, W)

img_feats = img_backbone(fake_imgs)

img_feats = img_neck(img_feats)

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