import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

register_coco_instances("coco_train", {}, 
                       "data/coco/annotations/instances_train2017.json", 
                       "data/coco/train2017")
register_coco_instances("coco_val", {}, 
                       "data/coco/annotations/instances_val2017.json", 
                       "data/coco/val2017")

cfg = get_cfg()
# use ResNet-101 FPN 3x config (1x config unaccessible)
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("coco_train",)
cfg.DATASETS.TEST = ("coco_val",)

cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 5000 
cfg.SOLVER.STEPS = (3000, 4000)

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
cfg.INPUT.MIN_SIZE_TRAIN = (800,)

cfg.OUTPUT_DIR = "fpn_output_101"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.SOLVER.CHECKPOINT_PERIOD = 1000

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True) 
trainer.train()