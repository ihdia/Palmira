# Weights for Initialization

COCO-Pretrained Model weights in the `init_weights` directory

## Weights Used

**[[`Mask RCNN R50-FPN-1x Link`](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl)]**

## Feel Free to use Other Weights

**[[`Detectron2 Model Zoo`](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn)]**

## Turning off Pre-Trained Weights

### Config Method

Modify the MODEL/WEIGHTS in config.yaml used

```yaml
MODEL:
  WEIGHTS: ''
```

### Command Line Method

```shell
python train_palmira.py \
    [... <other-args>] \
    MODEL.WEIGHTS ''
```