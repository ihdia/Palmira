#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine import default_argument_parser
from detectron2.engine import default_setup
from detectron2.engine import hooks
from detectron2.engine import launch
from detectron2.evaluation import CityscapesInstanceEvaluator
from detectron2.evaluation import CityscapesSemSegEvaluator
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import COCOPanopticEvaluator
from detectron2.evaluation import DatasetEvaluators
from detectron2.evaluation import LVISEvaluator
from detectron2.evaluation import PascalVOCDetectionEvaluator
from detectron2.evaluation import SemSegEvaluator
from detectron2.evaluation import verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA

from defgrid.config import add_defgrid_maskhead_config
# from hd.evaluator_perregion import HDEvaluator
# from hd.evaluator import HDEvaluator
from indiscapes_dataset import register_dataset
# from validation_hooks import EvalHook

register_dataset(combined_train_val=True)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow.

    They may not work for you, especially if you are working on a new
    research project. In that case you can write your own training loop.
    You can use "tools/plain_train_net.py" as an example.

    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.

        This uses the special metadata "evaluator_type" associated with
        each builtin dataset. For your own dataset, you can simply
        create an evaluator manually in your script and do not have to
        worry about the hacky if-else logic here.

        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ['sem_seg', 'coco_panoptic_seg']:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ['coco', 'coco_panoptic_seg', 'indiscapes']:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
            # evaluator_list.append(HDEvaluator(dataset_name))
        if evaluator_type == 'coco_panoptic_seg':
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == 'cityscapes_instance':
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), 'CityscapesEvaluator currently do not work with multiple machines.'
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == 'cityscapes_sem_seg':
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), 'CityscapesEvaluator currently do not work with multiple machines.'
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == 'pascal_voc':
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == 'lvis':
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                'no Evaluator for the dataset {} with the type {}'.format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger('detectron2.trainer')
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info('Running inference with test-time augmentation ...')
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, 'inference_TTA')
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + '_TTA': v for k, v in res.items()})
        return res

    def build_hooks(self):
        hooks = super().build_hooks()
        # hooks.insert(
        #     -1,
        #     EvalHook(
        #         self.cfg.TEST.EVAL_PERIOD,
        #         self.model,
        #         build_detection_test_loader(
        #             self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True)
        #         ),
        #     ),
        # )
        return hooks


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_defgrid_maskhead_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print('Command Line Args:', args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
