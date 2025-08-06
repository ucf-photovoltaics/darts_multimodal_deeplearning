import os
import json
import re
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from skimage import io

# Load in model

line = ['"COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"',2,0.001,1000, 100]

slurm_data = {'trained_models' : line[0][1:-1], 'batch_size' : int(line[1]), 
                  'learning_rate' : float(line[2]), 'iterations' : int(line[3]),
                  'output_dir' : int(line[4])}

def get_module_dicts(img_path) :
    files =  [x for x in os.listdir(img_path) if 'json' in x]

    dataset_dicts = []

    for file in files :
       
       # Load json image
        with open(os.path.join(img_path,file)) as f:
            img_annotation = json.load(f)
        
        # imageData is HUGE so we get rid of it to make our lives easier
        img_annotation['imageData'] = None

        # Store json data in detectron2 format

        record = {}

        record['file_name'] = os.path.join(img_path, img_annotation['imagePath'])
        record['height'] = img_annotation['imageHeight']
        record['width'] = img_annotation['imageWidth']
        record['image_id'] = re.split(r'[-.]', file)[5]

        # Store annotations in detectron2 format

        objs = []

        for obj in img_annotation['shapes']:
            x_points = np.array(obj['points'])[:,0]
            y_points = np.array(obj['points'])[:,1]

            poly = [p for x in obj['points'] for p in x]

            anno = {
                "bbox": [np.min(x_points) - 50, np.min(y_points)- 50, 
                np.max(x_points)+ 50, np.max(y_points) + 50],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0
            }

            objs.append(anno)

        record["annotations"] = objs

        dataset_dicts.append(record)
    return dataset_dicts

output_path = '/mnt/rstor/CSE_MSE_RXF131/cradle-members/sdle/wco3/sdle_repo/21-pv-multiscale/topics/image-processing/DeepLearning'

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(slurm_data['trained_models']))
cfg.DATASETS.TRAIN = ("module_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(slurm_data['trained_models'])  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = slurm_data['batch_size']  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = slurm_data['learning_rate']  # pick a good LR
cfg.SOLVER.MAX_ITER = slurm_data['iterations']    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (balloon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.OUTPUT_DIR = output_path + '/output-7thresh/output' + str(slurm_data['output_dir'])

cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + "/model_final.pth"  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

predictor = DefaultPredictor(cfg)

# Plot out some example results

test_path = '/home/wco3/CSE_MSE_RXF131/staging/sdle/pv-multiscale/test'

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

# Work on the images

img_path = '/mnt/rstor/CSE_MSE_RXF131/staging/sdle/pv-multiscale/MCCo-renamed-bottom/'
images = os.listdir(img_path)

img = io.imread(img_path + images[0])

output = predictor(img)