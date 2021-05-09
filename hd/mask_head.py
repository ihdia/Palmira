from typing import List

import torch
from detectron2.layers import cat
from detectron2.modeling.roi_heads import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads import MaskRCNNConvUpsampleHead
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from torch.nn import functional as F

# from hd.losses import WeightedHausdorffDistance


def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
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
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), 'Mask prediction must be square!'

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

    import matplotlib.pyplot as plt
    import numpy as np
    from hd.erosion import Dilation2d, Erosion2d
    from hd.losses import WeightedHausdorffDistance
    whd = WeightedHausdorffDistance(mask_side_len, mask_side_len, device=pred_mask_logits.device)
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction='mean')


    # whd()
    # instances[0]._fields['pred_classes'] = instances[0]._fields['gt_masks']
    # mask_rcnn_inference(pred_mask_logits, instances)
    # import numpy as np

    # def NormalizeData(data):
    #     return (data - np.min(data)) / (np.max(data) - np.min(data))

    # num_boxes_per_image = [len(i) for i in instances]
    # pred_mask_logits_split = pred_mask_logits.split(num_boxes_per_image, dim=0)
    # gt_masks_split = gt_masks.split(num_boxes_per_image, dim=0)

    # resized_pred_mask_bool = []
    # resized_gt_mask_bool = []

    # for instances_per_image, pred_mask_logits_per_image, gt_masks_per_image in zip(
    #     instances, pred_mask_logits_split, gt_masks_split
    # ):
    #     if len(instances_per_image) == 0:
    #         continue
    #     if not cls_agnostic_mask:
    #         resized_pred_mask_bool.append(
    #             retry_if_cuda_oom(paste_masks_in_image)(
    #                 pred_mask_logits_per_image,
    #                 instances_per_image.proposal_boxes,
    #                 instances_per_image._image_size,
    #             )
    #         )

    #         resized_gt_mask_bool.append(
    #             retry_if_cuda_oom(paste_masks_in_image)(
    #                 gt_masks_per_image,
    #                 instances_per_image.gt_boxes,
    #                 instances_per_image._image_size,
    #             )
    #         )

    # import cv2
    # cv2.imwrite("resized_pred_mask_logits[0][2].jpg", 255*resized_pred_mask_logits[0][2].to(torch.uint8).detach().cpu().numpy())
    # cv2.imwrite("resized_gt_mask[0][2].jpg", 255*resized_gt_mask[0][2].to(torch.uint8).detach().cpu().numpy())

    # print('Hi')

    # hausdorff_distance = []
    # for resized_pred_mask_bool_per_image, resized_gt_masks_bool_per_image in zip(
    #     resized_pred_mask_bool, resized_gt_mask_bool
    # ):
    #     whd = WeightedHausdorffDistance(
    #         resized_pred_mask_bool_per_image.shape[1],
    #         resized_pred_mask_bool_per_image.shape[2],
    #         device=resized_gt_masks_bool_per_image.device,
    #     )
    #     gt_batch = []
    #     for pred, gt in zip(
    #         resized_pred_mask_bool_per_image.detach().cpu().numpy(),
    #         resized_gt_masks_bool_per_image.detach().cpu().numpy(),
    #     ):
    #         footprint = generate_binary_structure(2, 1)
    #         gt_border = gt ^ binary_erosion(gt, structure=footprint, iterations=1)
    #         np.where(gt_border > 0)
    #         gt_batch.append()

    #         try:
    #             hausdorff_distance.append(hd(pred, gt))
    #         except:
    #             hausdorff_distance.append(100)

    #     whd(resized_pred_mask_bool_per_image, gt_border)

    # hausdorff_distance = sum(hausdorff_distance) / len(hausdorff_distance)

    # hausdorff_distance_95 = []
    # for resized_pred_mask_bool_per_image, resized_gt_masks_bool_per_image in zip(
    #     resized_pred_mask_bool, resized_gt_mask_bool
    # ):
    #     for pred, gt in zip(
    #         resized_pred_mask_bool_per_image.detach().cpu().numpy(),
    #         resized_gt_masks_bool_per_image.detach().cpu().numpy(),
    #     ):
    #         try:
    #             hausdorff_distance_95.append(hd95(pred, gt))
    #         except:
    #             hausdorff_distance_95.append(100)
    # hausdorff_distance_95 = sum(hausdorff_distance_95) / len(hausdorff_distance_95)

    # storage.put_scalar('mask_rcnn/hausdorff_distance', hausdorff_distance)
    # storage.put_scalar('mask_rcnn/hausdorff_distance_95', hausdorff_distance_95)

    return mask_loss


@ROI_MASK_HEAD_REGISTRY.register()
class HDMaskHead(MaskRCNNConvUpsampleHead):
    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        # x = F.sigmoid(x)  # Generate activations between 0 and 1 (For HD)
        # Conclusion: Dont apply sigmoid here. Do at at the HD Calculation. It hurts training.
        if self.training:
            return {'loss_mask': mask_rcnn_loss(x, instances, self.vis_period)}
        else:
            mask_rcnn_inference(x, instances)
            return instances
