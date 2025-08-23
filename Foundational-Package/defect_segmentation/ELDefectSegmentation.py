import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
img_proc_dir = project_root / "img_processing"
if img_proc_dir.as_posix() not in sys.path:
    sys.path.append(img_proc_dir.as_posix())

import cell_cropping
import cv2
import json
import matplotlib
import numpy as np
import os
import torch
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import transforms as t
from matplotlib import pyplot as plt
from PIL import Image

def build_custom_cmap():
    """Return the shared five‑colour colormap."""
    cmaplist = [
        (0.001462, 0.000466, 0.013866, 1.0),
        (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
        (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
        (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
        (1.0, 0.4980392156862745, 0.0, 1.0),
    ]
    return matplotlib.colors.LinearSegmentedColormap.from_list("Custom", cmaplist, len(cmaplist))


def crop_module_if_needed(path, index, *, rows=6, cols=10):
    """Auto/Manual crop a full module into cells when width > 2000 px."""
    try:
        cell_cropping.CellCropComplete(path, i=index, NumCells_y=rows, NumCells_x=cols, corners_get="auto")
    except cv2.error:
        print("Manual corner selection required. Click four corners, press 'c'; 'r' to reset.")
        cell_cropping.CellCropComplete(path, i=index, NumCells_y=rows, NumCells_x=cols, corners_get="manual")


def apply_defect_threshold(out_tensor, *, thresh):
    """Return int mask with 0 = no defect, 1‑4 defect classes."""
    soft = softmax(out_tensor)
    nodef = soft[0]
    nodef[nodef < thresh] = -1
    nodef[nodef >= thresh] = 0
    nodef = nodef.type(torch.int)
    def_idx = soft[1:].argmax(0).type(torch.int) + 1
    nodef[nodef == -1] = def_idx[nodef == -1]
    return nodef


def defect_stats(nodef):
    """Compute per‑class pixel fractions as a dict."""
    total_pix = torch.numel(nodef)
    return {
        "crack":        round(float(torch.count_nonzero(nodef == 1) / total_pix), 7),
        "contact":      round(float(torch.count_nonzero(nodef == 2) / total_pix), 7),
        "interconnect": round(float(torch.count_nonzero(nodef == 3) / total_pix), 7),
        "corrosion":    round(float(torch.count_nonzero(nodef == 4) / total_pix), 7),
    }

# editable parameters, set up to fit local file structure
##################################################
img_path = project_root / "output" / "CellsCropped"          # folder where input images are
model_path = current_file.parent / "models"                  # folder where model is stored
model_name = 'model_97.pth'                                  # trained model name
save_path = project_root / "output" / "SegmentationVisuals"  # where to save figures
defect_dir = project_root / "output" / "DefectPercentages"   # where to save defect % JSONs

defect_per = True                                            # turn on to save defect percentages
use_gpu = torch.cuda.is_available()                          # determines if GPU can be used to speed up computation
##################################################
################ model parameters ################
pre_model = 'deeplabv3_resnet50'      # backbone model was trained on
num_classes = 5                       # number of classes model trained on
threshold = .52                       # threshold for defect interpretation
aux_loss = True                       # loss type model trained with
##################################################

filelist = os.listdir(img_path)
if defect_per:
    os.makedirs(defect_dir, exist_ok=True)
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

# softmax layer for defect interpretation
softmax = torch.nn.Softmax(dim=0)

# this section loads in the weights of an already trained model
model = torchvision.models.segmentation.__dict__[pre_model](aux_loss=aux_loss, weights=None)

# changes last layer for output of appropriate class number
if pre_model == 'deeplabv3_resnet50' or pre_model == 'deeplabv3_resnet101':
    model.classifier = DeepLabHead(2048, num_classes)
else:
    num_ftrs_aux = model.aux_classifier[4].in_channels
    num_ftrs = model.classifier[4].in_channels
    model.aux_classifier[4] = torch.nn.Conv2d(num_ftrs_aux, num_classes, kernel_size=1)
    model.classifier[4] = torch.nn.Conv2d(num_ftrs, num_classes, kernel_size=1)

# model = model.cuda()
try:
    checkpoint = torch.load(model_path / model_name, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
except (FileNotFoundError, RuntimeError) as e:
    print("Checkpoint not found or corrupted – running with random weights.")
model.eval()

# transforms to put images through the model
trans = t.Compose([t.ToTensor(), t.Normalize(mean=.5, std=.2)])

# create the new map
cmap = build_custom_cmap()

if use_gpu:
    model = model.cuda()

i = 0

with torch.no_grad():
    # loops through every image in folder
    while i < len(filelist):
        # opens up and preps image, runs through model (RGB to benefit from pretrained model)
        if os.path.isdir(os.path.join(img_path, filelist[i])):
            i += 1
            continue
        im = Image.open(os.path.join(img_path, filelist[i])).convert('RGB')
        # meant to capture module images and crop them
        if im.size[0] > 2000:
            crop_module_if_needed(img_path / filelist[i], i)
            split_dir = img_path / f"Cell_Images{i}"
            split = [f"Cell_Images{i}/" + s for s in os.listdir(split_dir)]
            filelist.extend(split)
            i += 1
            continue
        img = trans(im).unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        output = model(img)['out']

        # threshold to determine defect vs. non-defect instead of softmax (custom for this model)
        nodef = apply_defect_threshold(output[0], thresh=threshold)

        if defect_per:
            # name is for saving json with defect percentage
            name = f"cell{i}"
            
            # counts stats of pixels/defect percentages
            stats = defect_stats(nodef)
            with open(defect_dir / f"{name}.json", "w") as fp:
                json.dump(stats, fp)

            output_defect_percent = sum(stats.values())

        if use_gpu:
            nodef = nodef.cpu()
            img = img.cpu()

        orig_img = (img * .2) + .5
        nodef = np.ma.masked_where(nodef == 0, nodef)

        # plots the original image next to prediction (with defect percentage)
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img[0][0].numpy(), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title('original image')
        
        plt.subplot(1, 2, 2)
        plt.imshow(orig_img[0][0], cmap='gray', vmin=0, vmax=1)
        plt.imshow(nodef, cmap=cmap, vmin=0, vmax=4, alpha=.3)
        plt.title('image + prediction')
        plt.tick_params(axis='both', labelsize=0, length=0)
        if defect_per:
            plt.xlabel(f"Defect Percentage: {output_defect_percent:.5f}")
        plt.savefig(os.path.join(save_path, str(i) + '.png'))  # comment back in to save figures
        # plt.show()
        plt.clf()
        
        
        i += 1
        # clear up memory
        if use_gpu:
            del nodef
            del img
            if defect_per:
                del output_defect_percent
