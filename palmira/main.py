
from defgrid.config import add_defgrid_maskhead_config
from detectron2.config import get_cfg
from predictor import VisualizationDemo


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_defgrid_maskhead_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def visulaization_demo(object):
    return VisualizationDemo(object)
