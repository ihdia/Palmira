from copy import deepcopy
from typing import List

import numpy as np
import torch
from detectron2.layers import Conv2d
from detectron2.layers import ConvTranspose2d
from detectron2.layers import ShapeSpec
from detectron2.layers import cat
from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from torch import nn
from torch.nn import functional as F

from defgrid.layers.DefGrid.diff_variance import LatticeVariance
from defgrid.models.deformable_grid import DeformableGrid
from defgrid.utils.matrix_utils import MatrixUtils


@ROI_MASK_HEAD_REGISTRY.register()
class DefGridHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super(DefGridHead, self).__init__()

        self.device = cfg.MODEL.DEVICE
        self.grid_size = cfg.MODEL.DEFGRID_MASK_HEAD.GRID_SIZE  # [20,20]
        self.grid_type = cfg.MODEL.DEFGRID_MASK_HEAD.GRID_TYPE  # dense_quad
        self.state_dim = cfg.MODEL.DEFGRID_MASK_HEAD.STATE_DIM  # 128
        self.out_dim = cfg.MODEL.DEFGRID_MASK_HEAD.OUT_DIM
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.sigma = cfg.MODEL.DEFGRID_MASK_HEAD.SIGMA
        self.mask_coef = cfg.MODEL.DEFGRID_MASK_HEAD.MASK_COEF

        self.w_variance = cfg.MODEL.DEFGRID_MASK_HEAD.W_VARIANCE
        self.w_area = cfg.MODEL.DEFGRID_MASK_HEAD.W_AREA
        self.w_laplacian = cfg.MODEL.DEFGRID_MASK_HEAD.W_LAPLACIAN
        self.w_reconstruct_loss = cfg.MODEL.DEFGRID_MASK_HEAD.W_RECONSTRUCT_LOSS

        self.matrix = MatrixUtils(1, self.grid_size, self.grid_type, self.device)

        self.model = DeformableGrid(cfg, self.device)

        self.to_three_channel = ConvTranspose2d(
            cfg.MODEL.ROI_MASK_HEAD.CONV_DIM, 3, kernel_size=2, stride=2, padding=0
        )
        # self.to_three_channel = Conv2d(cfg.MODEL.ROI_MASK_HEAD.CONV_DIM, 3, kernel_size=1, stride=1, padding=0)

        self.superpixel = LatticeVariance(
            28,
            28,
            sigma=self.sigma,
            device=self.device,
            add_seg=True,
            mask_coef=self.mask_coef,
        )

        self.mask_deconv = ConvTranspose2d(
            self.out_dim, self.out_dim, kernel_size=2, stride=2, padding=0
        )
        self.mask_predictor = Conv2d(
            self.out_dim, self.num_classes, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, instances: List[Instances]):
        self.input_dict = {}
        self.input_dict['net_input'] = x
        self.input_dict['crop_gt'] = instances

        n_batch = x.shape[0]

        base_point = self.matrix.init_point
        base_normalized_point_adjacent = self.matrix.init_normalized_point_adjacent
        base_point_mask = self.matrix.init_point_mask
        base_triangle2point = self.matrix.init_triangle2point
        base_area_mask = self.matrix.init_area_mask
        base_triangle_mask = self.matrix.init_triangle_mask

        self.input_dict['base_point'] = base_point.expand(n_batch, -1, -1)
        self.input_dict['base_normalized_point_adjacent'] = base_normalized_point_adjacent.expand(
            n_batch, -1, -1
        )
        self.input_dict['base_point_mask'] = base_point_mask.expand(n_batch, -1, -1)
        self.input_dict['base_triangle2point'] = base_triangle2point.expand(n_batch, -1, -1)
        self.input_dict['base_area_mask'] = base_area_mask.expand(n_batch, -1)
        self.input_dict['base_triangle_mask'] = base_triangle_mask.expand(n_batch, -1)
        self.input_dict['grid_size'] = np.max(self.grid_size)

        output = self.model(**self.input_dict)

        output['pred_points']
        gcn_pred_points = output['gcn_pred_points']

        mask_features = gcn_pred_points.reshape(
            -1, self.out_dim, self.grid_size[0], self.grid_size[1]
        )

        mask_features = F.relu(self.mask_deconv(mask_features))
        output['mask_logits'] = self.mask_predictor(mask_features)

        if self.training:
            loss_mask, loss_defgrid = self.mask_head_loss(
                output, instances, self.to_three_channel(x)
            )
            return {'loss_mask': loss_mask, 'loss_defgrid': loss_defgrid}
        else:
            mask_rcnn_inference(output['mask_logits'], instances)
            return instances

    def defgrid_loss(self, output, gt_masks, net_input):
        n_row_area_normalize = self.grid_size[0]
        n_column_area_normalize = self.grid_size[1]

        sub_batch_size = net_input.shape[0]
        variance = torch.zeros(sub_batch_size, device=self.device)
        laplacian_loss = torch.zeros(sub_batch_size, device=self.device)
        area_variance = torch.zeros(sub_batch_size, device=self.device)
        reconstruct_loss = torch.zeros(sub_batch_size, device=self.device)

        tmp_gt_mask = deepcopy(gt_masks.unsqueeze(1))
        tmp_gt_mask = tmp_gt_mask.long()
        gt_mask = self.gtmask2onehot(tmp_gt_mask).permute(0, 2, 3, 1)
        superpixel_ret = self.superpixel(
            grid_pos=output['pred_points'],
            img_fea=net_input[:, :3, ...].permute(0, 2, 3, 1),
            base_triangle2point=self.input_dict['base_triangle2point'],
            base_area_mask=self.input_dict['base_triangle_mask'],
            base_triangle_mask=self.input_dict['base_triangle_mask'],
            area_normalize=(n_row_area_normalize, n_column_area_normalize),
            semantic_mask=gt_mask,
            inference=False,
            grid_size=self.grid_size,
        )

        condition = superpixel_ret['condition']
        laplacian_loss += output['laplacian_energy']
        variance += superpixel_ret['variance']
        area_variance += superpixel_ret['area_variance']
        reconstruct_loss += superpixel_ret['reconstruct_loss']

        return condition, laplacian_loss, variance, area_variance, reconstruct_loss

    @staticmethod
    def gtmask2onehot(gtmask):
        batch_size, channel_num, h, w = gtmask.shape

        max_index = gtmask.max().long() + 1
        onehot = torch.zeros(batch_size, max_index, h, w, device=gtmask.device)
        onehot.scatter_(dim=1, index=gtmask.long(), value=1.0)

        return onehot

    def mask_head_loss(self, output, instances, net_input, vis_period: int = 0):
        """
        Compute the mask prediction loss defined in the Mask R-CNN paper.
        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.
        Returns:
            mask_loss (Tensor): A scalar tensor containing the loss.
        """
        pred_mask_logits = output['mask_logits']
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        total_num_masks = pred_mask_logits.size(0)
        mask_side_len = pred_mask_logits.size(2)
        assert pred_mask_logits.size(2) == pred_mask_logits.size(
            3
        ), 'Mask prediction must be square!'

        gt_classes = []
        gt_masks = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_classes.append(gt_classes_per_image)

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, mask_side_len
            ).to(device=pred_mask_logits.device)
            # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
            gt_masks.append(gt_masks_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)

        if cls_agnostic_mask:
            pred_mask_logits = pred_mask_logits[:, 0]
        else:
            indices = torch.arange(total_num_masks)
            gt_classes = cat(gt_classes, dim=0)
            pred_mask_logits = pred_mask_logits[indices, gt_classes]

        if gt_masks.dtype == torch.bool:
            gt_masks_bool = gt_masks
        else:
            # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
            gt_masks_bool = gt_masks > 0.5
        gt_masks = gt_masks.to(dtype=torch.float32)

        # Log the training accuracy (using gt classes and 0.5 threshold)
        mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
        mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
        num_positive = gt_masks_bool.sum().item()
        false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
            gt_masks_bool.numel() - num_positive, 1.0
        )
        false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

        storage = get_event_storage()
        storage.put_scalar('mask_rcnn/accuracy', mask_accuracy)
        storage.put_scalar('mask_rcnn/false_positive', false_positive)
        storage.put_scalar('mask_rcnn/false_negative', false_negative)
        if vis_period > 0 and storage.iter % vis_period == 0:
            pred_masks = pred_mask_logits.sigmoid()
            vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
            name = 'Left: mask prediction;   Right: mask GT'
            for idx, vis_mask in enumerate(vis_masks):
                vis_mask = torch.stack([vis_mask] * 3, axis=0)
                storage.put_image(name + f' ({idx})', vis_mask)

        mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction='mean')
        _, laplacian_loss, variance, area_variance, reconstruct_loss = self.defgrid_loss(
            output, gt_masks, net_input
        )

        defgrid_loss = (
            variance * self.w_variance
            + area_variance * self.w_area
            + laplacian_loss * self.w_laplacian
            + reconstruct_loss * self.w_reconstruct_loss
        )

        defgrid_loss = defgrid_loss.mean()

        return mask_loss, defgrid_loss
