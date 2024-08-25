# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmdet3d.registry import MODELS
from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from typing import Dict, List, Optional, Sequence
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None,
                 video_test_mode=False,
                 **kwargs):

        super(BEVFormer,
              self).__init__(pts_voxel_encoder, pts_middle_encoder,
                             pts_fusion_layer, img_backbone, pts_backbone,
                             img_neck, pts_neck, pts_bbox_head, img_roi_head,
                             img_rpn_head, train_cfg, test_cfg, init_cfg,
                             data_preprocessor, **kwargs)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    def extract_img_feat(self, img, input_metas, len_queue=None):
        """Extract features of images.

        Args:
            img: [bs, num_views, channel, H, W]
            len_queue: 
        """
        B = img.size(0)
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]

            if img.dim() == 5 and img.size(0) == 1:
                img = torch.squeeze(img, dim=1)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(
                    B // len_queue, len_queue, BN // B, C, H, W))
            else:
                img_feats_reshaped.append(
                    img_feat.view(B, BN // B, C, H, W))
        return img_feats_reshaped

    def extract_feat(self, imgs, batch_input_metas: List[dict],
                     len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(imgs, batch_input_metas, len_queue=len_queue)
        
        return img_feats

    def loss_imgs(self,
                  pts_feats,
                  batch_data_samples,
                  img_metas,
                  prev_bev=None):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .
                    gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                        boxes for each sample.
                    gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                        boxes of each sample
            prev_bev (torch.Tensor, optional): BEV features of previous frame.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        gt_bboxes_3d = batch_data_samples.gt_instances_3d.bboxes_3d
        gt_labels_3d = batch_data_samples.gt_instances_3d.labels_3d
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(
                imgs=imgs_queue,
                batch_input_metas=img_metas_list,
                len_queue=len_queue)
            
            for i in range(len_queue):
                img_metas = img_metas_list[i]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    def _forward(self, batch_inputs_dict, batch_data_samples):
        imgs = batch_inputs_dict.get("imgs", None)
        # batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = [item for item in batch_data_samples]
        len_queue = imgs.size(1)
        prev_img = imgs[:, :-1, ...]
        imgs = imgs[:, -1, ...]

        prev_img_metas = copy.deepcopy(batch_input_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        
        img_metas = [each[len_queue-1] for each in batch_input_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(imgs=imgs, batch_input_metas=img_metas)

        outs = self.pts_bbox_head(
            img_feats, img_metas, prev_bev)
        return outs
    
    def loss(self,
             batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             ):
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (bs, len_queue, num_cams, C, H, W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        imgs = batch_inputs_dict.get('imgs', None)
        len_queue = imgs.size(1)
        prev_img = imgs[:, :-1, ...]
        imgs = imgs[:, -1, ...]

        batch_input_metas = {}
        for queue_id in range(len_queue):
            batch_input_metas[queue_id] = [item.metainfo for item in batch_data_samples[queue_id]]
        prev_img_metas = copy.deepcopy(batch_input_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = batch_input_metas[len_queue-1]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(imgs=imgs, batch_input_metas=img_metas)
        losses = dict()
        losses_pts = self.loss_imgs(img_feats, batch_data_samples,
                                    img_metas, prev_bev)

        losses.update(losses_pts)
        return losses

    def predict(self, batch_inputs_dict,
                batch_data_samples, **kwargs):
        img_metas = [item.metainfo for item in batch_data_samples]
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = batch_inputs_dict.get('imgs', None)

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(imgs=img, batch_input_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
