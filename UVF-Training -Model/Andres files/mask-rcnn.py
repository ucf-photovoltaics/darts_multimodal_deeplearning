import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import imgaug

# Root directory of the Mask RCNN
ROOT_DIR = '/mnt/rstor/CSE_MSE_RXF131/cradle-members/sdle/wco3/sdle_repo/Mask_RCNN'

import warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

#Import Pycocotools
new_root = '/mnt/rstor/CSE_MSE_RXF131/cradle-members/sdle/wco3/sdle_repo/cocoapi/PythonAPI'
sys.path.append(new_root)
import pycocotools

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


