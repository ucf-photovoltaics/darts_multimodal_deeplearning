import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[1]            # Foundational‑Package/
img_proc_dir = project_root / "img_processing"
if img_proc_dir.as_posix() not in sys.path:
    sys.path.append(img_proc_dir.as_posix())

import importlib
cellCropping_mod = importlib.import_module('cellCropping')
sys.modules['cell_cropping'] = cellCropping_mod

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
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors

# Set up file paths
output_root      = project_root / "output"
img_path         = output_root / "CellsCropped"            # input folder
model_path       = current_file.parent / "models"          # model folder
model_name       = "model_97.pth"

unmasked_folder  = output_root / "LoadspotUnmasked"
masked_folder    = output_root / "LoadspotMasked"
visuals_folder   = output_root / "LoadspotVisuals"
defect_dir       = output_root / "LoadspotDefects"

for p in [unmasked_folder, masked_folder, visuals_folder, defect_dir]:
    os.makedirs(p, exist_ok=True)

# Model configuration
defect_per = True
use_gpu = torch.cuda.is_available()
pre_model = 'deeplabv3_resnet50'
num_classes = 5
threshold = 0.52
aux_loss = True

# Initialize model
softmax = torch.nn.Softmax(dim=0)
model = torchvision.models.segmentation.__dict__[pre_model](aux_loss=aux_loss, weights=None)

# Configure model layers
if pre_model == 'deeplabv3_resnet50' or pre_model == 'deeplabv3_resnet101':
    model.classifier = DeepLabHead(2048, num_classes)
else:
    num_ftrs_aux = model.aux_classifier[4].in_channels
    num_ftrs = model.classifier[4].in_channels
    model.aux_classifier[4] = torch.nn.Conv2d(num_ftrs_aux, num_classes, kernel_size=1)
    model.classifier[4] = torch.nn.Conv2d(num_ftrs, num_classes, kernel_size=1)

# Load model weights
try:
    checkpoint = torch.load(model_path / model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
except (FileNotFoundError, RuntimeError, KeyError):
    print('Checkpoint not found or corrupted – running with random weights.')
model.eval()

# Define transformations
trans = t.Compose([t.ToTensor(), t.Normalize(mean=0.5, std=0.2)])

# Create custom colormap
cmaplist = [
    (0.001462, 0.000466, 0.013866, 1.0),
    (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
    (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
    (1.0, 0.4980392156862745, 0.0, 1.0)
]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom', cmaplist, len(cmaplist))

# Move model to GPU if available
if use_gpu:
    model = model.cuda()

i = 0
with torch.no_grad():
    # Process all images in the input folder
    filelist = os.listdir(img_path)
    
    while i < len(filelist):
        # Skip directories
        if os.path.isdir(os.path.join(img_path, filelist[i])):
            i += 1
            continue
            
        # Load and process image
        im = Image.open(os.path.join(img_path, filelist[i])).convert('RGB')
        
        # Handle large module images
        if im.size[0] > 2000:
            try:
                cell_cropping.CellCropComplete(os.path.join(img_path, filelist[i]), 
                                            i=i, NumCells_y=6, NumCells_x=12, corners_get='auto')
            except cv2.error:
                print('This module needs manual corner finding. Click each of the four corners, then press \'c\'. '
                      'In case of mistake, please press \'r\' to reset corners.')
                cell_cropping.CellCropComplete(os.path.join(img_path, filelist[i]), 
                                            i=i, NumCells_y=6, NumCells_x=12, corners_get='manual')
            
            # Add cropped images to processing list
            split = os.listdir(os.path.join(img_path, 'Cell_Images' + str(i)))
            split = [os.path.join('Cell_Images' + str(i) + '/', s) for s in split]
            filelist.extend(split)
            i += 1
            continue
        
        # Prepare image for model
        img = trans(im).unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        
        # Run prediction
        output = model(img)['out']
        
        # Process predictions
        soft = softmax(output[0])
        nodef = soft[0]
        nodef[nodef < threshold] = -1
        nodef[nodef >= threshold] = 0
        nodef = nodef.type(torch.int)
        def_idx = soft[1:].argmax(0).type(torch.int)
        def_idx = def_idx + 1
        nodef[nodef == -1] = def_idx[nodef == -1]
        
        # Calculate defect percentages if enabled
        if defect_per:
            name = 'cell' + str(i)
            output_pix = torch.count_nonzero(nodef)
            total_pix = torch.numel(nodef.detach())
            output_defect_percent = torch.div(output_pix.type(torch.float), total_pix)
            
            if use_gpu:
                output_defect_percent = output_defect_percent.cpu()
                
            crack_portion = torch.div(torch.count_nonzero(nodef == 1), total_pix)
            contact_portion = torch.div(torch.count_nonzero(nodef == 2), total_pix)
            interconnect_portion = torch.div(torch.count_nonzero(nodef == 3), total_pix)
            corrosion_portion = torch.div(torch.count_nonzero(nodef == 4), total_pix)
            
            # Save defect percentages to JSON
            defect_percentages = {
                'crack': round(float(crack_portion), 7),
                'contact': round(float(contact_portion), 7),
                'interconnect': round(float(interconnect_portion), 7),
                'corrosion': round(float(corrosion_portion), 7)
            }
            with open(os.path.join(defect_dir, name + '.json'), 'w') as fp:
                json.dump(defect_percentages, fp)
        
        # Move tensors to CPU if needed
        if use_gpu:
            nodef = nodef.cpu()
            img = img.cpu()
        
        orig_img = (img * 0.2) + 0.5
        nodef = np.ma.masked_where(nodef == 0, nodef)
        
        # Save unmasked image
        unmasked_array = orig_img[0][0].numpy()
        unmasked_filename = os.path.splitext(filelist[i])[0] + '.png'
        unmasked_path = os.path.join(unmasked_folder, unmasked_filename)
        cv2.imwrite(unmasked_path, (unmasked_array * 255).astype(np.uint8))

        # Create masked-only visualization
        plt.figure(figsize=(10, 5))
        plt.imshow(orig_img[0][0].numpy(), cmap='gray', vmin=0, vmax=1)
        plt.imshow(nodef, cmap=cmap, vmin=0, vmax=4, alpha=0.3)
        plt.axis('off')
        masked_path = os.path.join(masked_folder, unmasked_filename)
        plt.savefig(masked_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
        # Create combined visualization
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].imshow(orig_img[0][0].numpy(), cmap='gray', vmin=0, vmax=1)
        ax[0].axis('off')
        ax[0].set_title('original image')
        ax[1].imshow(orig_img[0][0].numpy(), cmap='gray', vmin=0, vmax=1)
        ax[1].imshow(nodef, cmap=cmap, vmin=0, vmax=4, alpha=0.3)
        ax[1].axis('off')
        ax[1].set_title('image + prediction')
        if defect_per:
            ax[1].set_xlabel("Defect Percentage: " + str(output_defect_percent.numpy().round(5)))
        visual_path = os.path.join(visuals_folder, unmasked_filename)
        plt.savefig(visual_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        i += 1
        
        # Clean up memory
        if use_gpu:
            del nodef
            del img
        if defect_per:
            del output_defect_percent