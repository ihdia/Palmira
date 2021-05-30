# Model Zoo

## Ablative Variants

### Deformable Backbone

Remove defgrid mask head by commenting out the `add_defgrid_maskhead_config(cfg)` from the `def setup(args)`
in `train_net_palmira.py`

```bash
python train_palmira.py \
    --config-file configs/dconv/dconv_c3-c5.yaml \
    --num-gpus 4
```

To toggle other layers of the ResNet, edit this section of the config.yaml

```yaml
MODEL:
  RESNETS:
    DEFORM_ON_PER_STAGE: [ False, True, True, True ] # on Res3, Res4, Res5
```

### Vanilla Mask RCNN

Remove defgrid mask head by commenting out the `add_defgrid_maskhead_config(cfg)` from the `def setup(args)`
in `train_net_palmira.py`

```bash
python train_palmira.py \
    --config-file configs/mrcnn/vanilla_mrcnn.yaml \
    --num-gpus 4
```

## Other Baselines

Please refer to respective folders for code, train scripts, configs, and the README for usage and installation instructions.

# References

| Model Name |                                   Official Code Repo                                  |
|:----------:|:-------------------------------------------------------------------------------------:|
|  CondInst  |      [Code](https://github.com/aim-uofa/AdelaiDet/tree/master/configs/CondInst)       |
| PointRend  | [Code](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend) |
|   BMRCNN   |       [Code](https://github.com/hustvl/BMaskR-CNN/tree/master/projects/BMaskR-CNN)    |
|    DETR    |            [Code](https://github.com/facebookresearch/detr/tree/master/d2)            |

```bibtex
@misc{carion2020endtoend,
      title={End-to-End Object Detection with Transformers}, 
      author={Nicolas Carion and Francisco Massa and Gabriel Synnaeve and Nicolas Usunier and Alexander Kirillov and Sergey Zagoruyko},
      year={2020},
      eprint={2005.12872},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@inproceedings{tian2020conditional,
  title     =  {Conditional Convolutions for Instance Segmentation},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao},
  booktitle =  {Proc. Eur. Conf. Computer Vision (ECCV)},
  year      =  {2020}
}
```

```bibtex
@article{ChengWHL20,
  title={Boundary-preserving Mask R-CNN},
  author={Tianheng Cheng and Xinggang Wang and Lichao Huang and Wenyu Liu},
  booktitle={ECCV},
  year={2020}
}
```

```bibtex
@InProceedings{kirillov2019pointrend,
  title={{PointRend}: Image Segmentation as Rendering},
  author={Alexander Kirillov and Yuxin Wu and Kaiming He and Ross Girshick},
  journal={ArXiv:1912.08193},
  year={2019}
}
```