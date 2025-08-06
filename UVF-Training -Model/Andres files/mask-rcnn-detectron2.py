import os
import sys
import re
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import json
import cv2
from skimage import io
import time
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


#pip install 'git+https://github.com/facebookresearch/detectron2.git'
import detectron2
from detectron2.structures import BoxMode
from detectron2.data import DatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
setup_logger()

random.seed(12345)

# Convert labelme json to detectron2 format

img_path = "/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Case Western Segmentation/training"

def get_module_dicts(img_path):
    failed_dir = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs/failed"
    os.makedirs(failed_dir, exist_ok=True)

    files = [x for x in os.listdir(img_path) if x.endswith(".json")]
    dataset_dicts = []

    for file in files:
        full_json_path = os.path.join(img_path, file)

        try:
            with open(full_json_path) as f:
                img_annotation = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON decode failed for {file}: {e}")
            shutil.move(full_json_path, os.path.join(failed_dir, file))
            continue

        image_file = os.path.join(img_path, img_annotation["imagePath"])
        if not os.path.exists(image_file):
            print(f"[WARNING] Skipping {file}: Image file '{image_file}' does not exist.")
            shutil.move(full_json_path, os.path.join(failed_dir, file))
            continue

        try:
            img = cv2.imread(image_file)
            if img is None:
                raise ValueError("cv2.imread returned None")
        except Exception as e:
            print(f"[WARNING] Corrupt image for {file}: {e}")
            shutil.move(full_json_path, os.path.join(failed_dir, file))
            shutil.move(image_file, os.path.join(failed_dir, os.path.basename(image_file)))
            continue

        img_annotation['imageData'] = None

        record = {
            "file_name": image_file,
            "height": img_annotation['imageHeight'],
            "width": img_annotation['imageWidth'],
            "image_id": re.split(r'[-.]', file)[5],
        }

        objs = []
        for obj in img_annotation['shapes']:
            x_points = np.array(obj['points'])[:, 0]
            y_points = np.array(obj['points'])[:, 1]
            poly = [p for x in obj['points'] for p in x]

            anno = {
                "bbox": [np.min(x_points) - 50, np.min(y_points) - 50,
                         np.max(x_points) + 50, np.max(y_points) + 50],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0
            }
            objs.append(anno)

        record["annotations"] = objs
        dataset_dicts.append(record)

    print(f"[INFO] Number of valid annotations loaded: {len(dataset_dicts)}")
    return dataset_dicts

DatasetCatalog.clear()
DatasetCatalog.register("module_train", lambda d = img_path: get_module_dicts(d))
DatasetCatalog.register("module_test", lambda d = '/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Case Western Segmentation/test': get_module_dicts(d))
MetadataCatalog.get("module_train").set(thing_classes=["module"])
MetadataCatalog.get("module_test").set(thing_classes=["module"])

# Test to see if it worked
# dataset_dicts = get_module_dicts(img_path)
# print(f"Number of annotations loaded: {len(dataset_dicts)}")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(os.path.join(img_path, d["file_name"]))
#     visualizer = Visualizer(img[:, :, ::-1], scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
#     io.imshow(out.get_image()[:, :, ::-1])  
#     plt.show()

# Use a pre-trained model to work with the data
# "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" base use

cfg = get_cfg()
cfg.OUTPUT_DIR = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
cfg.DATASETS.TRAIN = ("module_train",)
cfg.DATASETS.TEST = ("module_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 100    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (balloon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

t = time.time()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

train_time = time.time() - t
# Evaluate Trained Model

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

test_path = "/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Case Western Segmentation/test"

dataset_dicts = get_module_dicts(test_path)
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(os.path.join(test_path, d["file_name"]))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    io.imshow(out.get_image()[:, :, ::-1])  
    plt.show()

# Evaluate the model

evaluator = COCOEvaluator("module_test", output_dir="/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs/json/")
val_loader = build_detection_test_loader(cfg, "module_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))

# Evaluate Trained Model on other datasets

cfg = get_cfg()
cfg.OUTPUT_DIR = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs"  # <--- add output dir here too
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("module_train",)
cfg.DATASETS.TEST = ("module_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # Load your newly trained model!
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

predictor = DefaultPredictor(cfg)


test_path = "/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Case Western Segmentation/test"

dataset_dicts = get_module_dicts(test_path)
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(os.path.join(test_path, d["file_name"]))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    io.imshow(out.get_image()[:, :, ::-1])  
    plt.show()

# Evaluate the model
evaluator = COCOEvaluator("module_test", output_dir="/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs/json/")
val_loader = build_detection_test_loader(cfg, "module_test")
print(inference_on_dataset(predictor.model, val_loader, evaluator))