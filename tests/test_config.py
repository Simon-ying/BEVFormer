_base_ = [
    'mmdet::_base_/default_runtime.py'
]

import os
import sys
project_path = "/home/yingzhuoye/ws/github_files/BEVFormer"
sys.path.append(project_path)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

_dim_ = 64
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 4
bev_h_ = 50
bev_w_ = 50
queue_length = 2 # each sequence contains `queue_length` frames.

custom_imports = dict(imports=['projects.mmdet3d_plugin.bevformer', 'projects.mmdet3d_plugin.core', 'projects.mmdet3d_plugin.models.layers'], allow_failed_imports=True)

img_backbone=dict(
    type='mmdet.ResNet',
    depth=101,
    num_stages=4,
    out_indices=(1,2,3,),
    frozen_stages=1,
    # "BN2d and DCNv2 in mmcv, fallback_on_stride used in resnet.py in mmdet"
    norm_cfg=dict(type='BN2d', requires_grad=False),
    norm_eval=True,
    style='caffe',
    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
    stage_with_dcn=(False, False, True, True))

img_neck=dict(
    type='mmdet.FPN',
    in_channels=[512, 1024, 2048],
    out_channels=_dim_,
    start_level=0,
    add_extra_convs='on_output',
    num_outs=4,
    relu_before_extra_convs=True)

transformer=dict(
    type='PerceptionTransformer',
    rotate_prev_bev=True,
    use_shift=True,
    use_can_bus=True,
    embed_dims=_dim_,
    rotate_center=[bev_h_//2, bev_w_//2],
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
            embed_dims=_dim_,
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
                num_levels=1,
            ),
            ffn_cfg=dict(
                embed_dims=_dim_,
                feedforward_channels=_ffn_dim_,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ))))

pts_bbox_head=dict(
    type='mmdet.BEVFormerHead',
    bev_h=bev_h_,
    bev_w=bev_w_,
    num_query=900,
    num_classes=10,
    in_channels=_dim_,
    sync_cls_avg_factor=True,
    with_box_refine=True,
    as_two_stage=False,
    transformer=transformer,
    bbox_coder=dict(
        type='NMSFreeCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=300,
        voxel_size=voxel_size,
        num_classes=10),
    positional_encoding=dict(
        type='LearnedPositionalEncoding3D',
        num_feats=_pos_dim_,
        row_num_embed=bev_h_,
        col_num_embed=bev_w_,
        ),
    loss_cls=dict(
        type='mmdet.FocalLoss',
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0),
    loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
    loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0))

model = dict(
    type='BEVFormer',
    use_grid_mask=True,
    video_test_mode=True,
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type='mmdet.BEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=transformer,
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding3D',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))