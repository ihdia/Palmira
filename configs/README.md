### Deformable Backbone

Remove defgrid mask head by commenting out the `add_defgrid_maskhead_config(cfg)` from the basic setup
in `train_net_palmira.py`

```bash
Python train_palmira.py --config-file  configs/dconv/dconv_c3-c5.yaml num-gpus 4 --resume
```

### Vanilla Mask RCNN

Defgrid mask needs to be removed here as well.

```bash
Python train_palmira.py --config-file  configs/mrcnn/vanilla_mrcnn.yaml num-gpus 4 --resume
```