_base_ = [
    'mmdet::_base_/default_runtime.py'
]

import os
import sys
project_path = "/home/yingzhuoye/ws/github_files/BEVFormer"
sys.path.append(project_path)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
queue_length = 4 # each sequence contains `queue_length` frames.

custom_imports = dict(imports=['projects.mmdet3d_plugin.bevformer'], allow_failed_imports=True)

transformer=dict(
    type='PerceptionTransformer',
    rotate_prev_bev=True,
    use_shift=True,
    use_can_bus=True,
    embed_dims=_dim_,
    encoder=dict(
        type='BEVFormerEncoder',
        num_layers=6,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        return_intermediate=False,
        transformerlayers=dict(
            type='BEVFormerLayer',
            attn_cfgs=[
                dict(
                    type='TemporalSelfAttention',
                    embed_dims=_dim_,
                    num_levels=1),
                dict(
                    type='SpatialCrossAttention',
                    pc_range=point_cloud_range,
                    deformable_attention=dict(
                        type='MSDeformableAttention3D',
                        embed_dims=_dim_,
                        num_points=8,
                        num_levels=_num_levels_),
                    embed_dims=_dim_,
                )
            ],
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                'ffn', 'norm'))),
    decoder=dict(
        type='DetectionTransformerDecoder',
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type='mmdet.DetrTransformerDecoderLayer',
            self_attn_cfg=dict(
                type='MultiheadAttention',
                embed_dims=_dim_,
                num_heads=8,
                dropout=0.1
            ),
            cross_attn_cfg=dict(
                type='CustomMSDeformableAttention',
                embed_dims=_dim_,
                num_levels=1
            ),
            ffn_cfg=dict(
                feedforward_channels=_ffn_dim_,
                ffn_drop=0.1,
            ))))