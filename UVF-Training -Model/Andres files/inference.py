# Author: Andres Ojeda Sainz
# Date: 6/09/25
# Script that takes pre-trained model to run inference and print results.

import os
import cv2
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo

# === Paths ===
model_path = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs/model_final.pth"
inference_path = "/mnt/vstor/CSE_MSE_RXF131/staging/sdle/pv-multiscale/UCF-Image-Data/M55/UVF/F0201"
output_path = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs/inference_results"

os.makedirs(output_path, exist_ok=True)

# === Load trained model ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

predictor = DefaultPredictor(cfg)

# === Run inference ===
for fname in os.listdir(inference_path):
    if not (fname.endswith(".tiff") or fname.endswith(".jpg") or fname.endswith(".jpeg")):
        continue

    full_path = os.path.join(inference_path, fname)
    im = cv2.imread(full_path)

    if im is None:
        print(f"[WARNING] Skipping unreadable image: {fname}")
        continue

    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    base_name, _ = os.path.splitext(fname)
    save_path = os.path.join(output_path, f"{base_name}_pred.png")
    Image.fromarray(out.get_image()[:, :, ::-1]).save(save_path)
    print(f"[INFO] Saved: {save_path}")
