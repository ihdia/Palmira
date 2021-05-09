from detectron2.config import CfgNode as CN


def add_hd_maskhead_config(cfg):
    cfg.MODEL.HD_MASK_HEAD = CN()
    cfg.MODEL.HD_MASK_HEAD.HD_LOSS = True

    cfg.TEST.REPORT_HD_METRIC = True
