from mmdet.registry import MODELS
from mmengine import Config
cfg = Config.fromfile("test_config.py")

transformer = MODELS.build(cfg.transformer)
print(transformer)