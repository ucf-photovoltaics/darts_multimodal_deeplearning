
# Dawn Balaschak 7/12/2025
# UVF model training and evaluation using Detectron2 for multimodal integration

import os
import random
import cv2
import json

# --- Detectron2 Setup ---
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from pycocotools.coco import COCO

# --- Paths ---
base_directory = os.getcwd()
output_dir = os.path.join(base_directory, "outputs")
os.makedirs(output_dir, exist_ok=True)

train_json = os.path.join(base_directory, "Annotated/split_data/train.json")
test_json = os.path.join(base_directory, "Annotated/split_data/test.json")
train_img_dir = os.path.join(base_directory, "Annotated/split_data/train")
test_img_dir = os.path.join(base_directory, "Annotated/split_data/test")

# --- Register Datasets ---
register_coco_instances("uvf_train", {}, train_json, train_img_dir)
register_coco_instances("uvf_test", {}, test_json, test_img_dir)
MetadataCatalog.get("uvf_train").thing_classes = [
    "square", "ring", "crack", "bright_crack", "hotspot",
    "finger_corrosion", "near_busbar", "busbar_crack", "misc",
    "unused_1", "unused_2"
]

# --- Configuration ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("uvf_train",)
cfg.DATASETS.TEST = ("uvf_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
cfg.OUTPUT_DIR = output_dir
cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# --- Choose model weight source ---
trained_weights_path = "/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/trained_models/model_final.pth"
if os.path.exists(trained_weights_path):
    cfg.MODEL.WEIGHTS = trained_weights_path
else:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# --- Data Validation (optional toggle) ---
if False:
    coco = COCO(train_json)
    missing_train = [img['file_name'] for img in coco.loadImgs(coco.getImgIds()) if not os.path.exists(os.path.join(train_img_dir, img['file_name']))]
    print("Missing training images:", missing_train)

    coco = COCO(test_json)
    missing_test = [img['file_name'] for img in coco.loadImgs(coco.getImgIds()) if not os.path.exists(os.path.join(test_img_dir, img['file_name']))]
    print("Missing test images:", missing_test)

# --- Train or Evaluate Toggle ---
TRAIN_MODE = True

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)

if TRAIN_MODE:
    trainer.train()
else:
    evaluator = COCOEvaluator("uvf_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "uvf_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)

# --- Visualize Predictions ---
predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get("uvf_test")
metadata = MetadataCatalog.get("uvf_test")

for d in random.sample(dataset_dicts, 5):
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_path = os.path.join(cfg.OUTPUT_DIR, f"pred_{os.path.basename(d['file_name'])}")
    cv2.imwrite(out_path, out.get_image()[:, :, ::-1])

print("Training/Evaluation completed. Outputs saved to:", cfg.OUTPUT_DIR)
