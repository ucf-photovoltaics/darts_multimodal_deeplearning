# Author: Andres Ojeda Sainz
# Date: 6/09/25
# Script that takes pre-trained model to run inference on segmented PV cells.

import os
import cv2
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo

# === Paths ===
model_path = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/model_final.pth"
inference_path = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs/segmented_cells"
output_path = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/outputs/inference_results"

os.makedirs(output_path, exist_ok=True)

# === Load trained model ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # One class (e.g., "defect")

predictor = DefaultPredictor(cfg)

# === Run inference ===
for fname in os.listdir(inference_path):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".tiff")):
        continue

    full_path = os.path.join(inference_path, fname)
    im = cv2.imread(full_path)

    if im is None:
        print(f"[WARNING] Skipping unreadable image: {fname}")
        continue

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")
    print(f"[INFO] {fname} → {len(instances)} predictions")

    v = Visualizer(im[:, :, ::-1], scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(instances)

    base_name = os.path.splitext(fname)[0]
    save_path = os.path.join(output_path, f"{base_name}_pred.png")
    Image.fromarray(out.get_image()[:, :, ::-1]).save(save_path)
    print(f"[INFO] Saved: {save_path}")