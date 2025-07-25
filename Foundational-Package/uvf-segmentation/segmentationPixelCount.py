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
import csv

# editable parameters, set up to fit local file structure
##################################################
img_path = '/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/UCF-EL-Defect/M55 EL Data/3x12/Cells'             # folder where images are
model_path = 'models'                # folder where model is stored
model_name = 'model_97.pth'           # trained model name
save_path = '//mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/UCF-EL-Defect/M55 EL Data/3x12/PixelVisuals'                 # location to save figures
defect_dir = '/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/UCF-EL-Defect/M55 EL Data/3x12/PixelDefects'    # location to save defect percentage jsons
csv_path = 'analysisM55.csv'            # location to save CSV analysis
defect_per = True                    # turn on if you want to see defect percentages
use_gpu = torch.cuda.is_available()   # determines if GPU can be used to speed up computation
##################################################
################ model parameters ################
pre_model = 'deeplabv3_resnet50'      # backbone model was trained on
num_classes = 5                       # number of classes model trained on
threshold = .52                       # threshold for defect interpretation
aux_loss = True                      # loss type model trained with
##################################################

# Create CSV header
csv_columns = ['filename', 'width_pixels', 'height_pixels', 'total_pixels',
               'crack_pixels', 'contact_pixels', 'interconnect_pixels', 'corrosion_pixels']
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)

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
checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# transforms to put images through the model
trans = t.Compose([t.ToTensor(), t.Normalize(mean=.5, std=.2)])

# create custom colormap for image visualizations [Black, Red, Blue, Purple, Orange]
cmaplist = [(0.001462, 0.000466, 0.013866, 1.0),
            (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),
            (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),
            (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0),
            (1.0, 0.4980392156862745, 0.0, 1.0)]

# create the new map
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom', cmaplist, len(cmaplist))

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
        width, height = im.size
        
        # meant to capture module images and crop them
        if width > 2000:
            # can switch to corners_get='manual' for manual cropping, in case of fail will automatically switch
            try:
                cell_cropping.CellCropComplete(os.path.join(img_path, filelist[i]), i=i, NumCells_y=6, NumCells_x=12, corners_get='auto')
            except cv2.error:
                print('This module needs manual corner finding. Click each of the four corners, then press \'c\'. '
                      'In case of mistake, please press \'r\' to reset corners.')
                cell_cropping.CellCropComplete(os.path.join(img_path, filelist[i]), i=i, NumCells_y=6, NumCells_x=12, corners_get='manual')

            split = os.listdir(os.path.join(img_path, 'Cell_Images' + str(i)))
            split = [os.path.join('Cell_Images' + str(i) + '/', s) for s in split]
            filelist.extend(split)
            i += 1
            continue
        
        img = trans(im).unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        output = model(img)['out']

        # threshold to determine defect vs. non-defect instead of softmax (custom for this model)
        soft = softmax(output[0])
        nodef = soft[0]
        nodef[nodef < threshold] = -1
        nodef[nodef >= threshold] = 0
        nodef = nodef.type(torch.int)
        def_idx = soft[1:].argmax(0).type(torch.int)
        def_idx = def_idx + 1
        nodef[nodef == -1] = def_idx[nodef == -1]

        total_pixels = width * height
        crack_pixels = torch.count_nonzero(nodef == 1).item()
        contact_pixels = torch.count_nonzero(nodef == 2).item()
        interconnect_pixels = torch.count_nonzero(nodef == 3).item()
        corrosion_pixels = torch.count_nonzero(nodef == 4).item()

        # Write CSV row
        csv_row = [
            filelist[i],
            width,
            height,
            total_pixels,
            crack_pixels,
            contact_pixels,
            interconnect_pixels,
            corrosion_pixels
        ]
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_row)

        if defect_per:
            # name is for saving json with defect percentage
            name = 'cell' + str(i)

            # counts stats of pixels/defect percentages
            output_pix = torch.count_nonzero(nodef)
            total_pix = torch.numel(nodef.detach())

            output_defect_percent = torch.div(output_pix.type(torch.float), total_pix)
            if use_gpu:
                output_defect_percent = output_defect_percent.cpu()

            crack_portion = torch.div(torch.count_nonzero(nodef == 1), total_pix)
            contact_portion = torch.div(torch.count_nonzero(nodef == 2), total_pix)
            interconnect_portion = torch.div(torch.count_nonzero(nodef == 3), total_pix)
            corrosion_portion = torch.div(torch.count_nonzero(nodef == 4), total_pix)

            # creates json to save defect percentage per class category
            defect_percentages = {'crack': round(float(crack_portion), 7), 'contact': round(float(contact_portion), 7),
                                  'interconnect': round(float(interconnect_portion), 7),
                                  'corrosion': round(float(corrosion_portion), 7)}

            with open(os.path.join(defect_dir, name + '.json'), 'w') as fp:
                json.dump(defect_percentages, fp)

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
            plt.xlabel("Defect Percentage: " + str(output_defect_percent.numpy().round(5)))

        plt.savefig(os.path.join(save_path, str(i) + '.png'))  # comment back in to save figures
        plt.show()

        plt.clf()
        i += 1
        # clear up memory
        if use_gpu:
            del nodef
            del img
            if defect_per:
                del output_defect_percent