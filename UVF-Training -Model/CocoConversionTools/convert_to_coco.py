                                                     # changelog
# 07-19-2025 Dawn Balaschak 
#   Details: 
#   vgg to coco format conversion for the UVF training system (idk why this was available to someone sooner 

#   Inputs: 
#       directory paths for where your images are, 
#       the csv_annotation path, 
#       the output directory wanted 
#       list of desired skipped images, can be generated if it does not exist already
#            
#   Outputs: 5 files, list of the follow
#       broken annotation image names
#       broken annotations
#       missing images from the image_dir
#       suggested skip list that can be used to rerun the program without corrupt or missing data
#       output.json <--- actual coco formatted json 
  
#   Issues:
#      not currently set for figuring out the annotations that are not using rect, polygon, polyline, those get suggested to be skipped
#   Suggestions 
#       Have it run twice, once to generate the suggested skip list, and then once again with the suggested skiplist for non programming efficient users
#       Have it create a folder with the images and output.json as a pair 
#       
#       
#   Notes:
#       Using the main annotations.csv from the base of Sina's Classification, it had numerous issues within the annotations csv, it needs to be corrected
#       I manually adjusted a few of the smaller typos that I found but this comes from using a very poor annotation software
#       I recommend you look into usinv CVAT, as that it is more user friendly, and can easily convert to numerous types of annotations 





import os
import json
import cv2
from matplotlib import category
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd         # added pandas for dataframe use ing 
from PIL import Image


base_directory = os.getcwd()


# directories adjust to use base directory solutions  use copy realtive path
csv_path = r"CocoConversionTools/uvf_training_annotations_v01.csv"
images_dir = r"Annotated/Images"
output_dir = r"CocoConversionTools/Outputs"

csv_path = os.path.join(base_directory, csv_path)
images_dir = os.path.join(base_directory, images_dir)
output_dir = os.path.join(base_directory, output_dir)

os.makedirs(output_dir, exist_ok=True)

# this can be copied from the the coco_conversion_suggested_skip_list. 
# Essentially it makes it so that the images that are either not found, or have corrupted annotations do not get included into the 
# coco conversion THIS IS DONE MANUALLY
custom_suggested_skip_list = []

custom_suggested_skip_list = ["VCAD-0009_cell_06_INDOOR.jpg", "PSEL-2534_cell_10_INDOOR.jpg", "PSEL-2537_cell_10_INDOOR.jpg", "PSEL-2538_cell_09.jpg", "PSEL-2538_cell_27_INDOOR.jpg", "RA02-0020_cell_09.jpg", "VCAD-0012_cell_06_INDOOR.jpg", "RA17-0032_cell_01.jpg", "PSEL-2534_cell_05.jpg", "PSEL-2536_cell_58_INDOOR.jpg", "PSEL-2536_cell_43.jpg", "VCAD-0009_cell_06.jpg", "PSEL-2534_cell_06.jpg", "VCAD-0009_cell_52.jpg", "VCAD-0012_cell_30.jpg", "VCAD-0008_cell_05_INDOOR.jpg", "VCAD-0009_cell_52_INDOOR.jpg", "PSEL-2530_cell_19_INDOOR.jpg", "PSEL-2537_cell_53.jpg", "VCAD-0009_cell_37.jpg", "RA02-0004_cell_28.jpg", "VCAD-0009_cell_10.jpg", "VCAD-0016_cell_34_INDOOR.jpg", "RA17-0032_cell_39.jpg", "PSEL-2538_cell_08_INDOOR.jpg", "PSEL-2538_cell_33_INDOOR.jpg", "PSEL-2529_cell_65.jpg", "PSEL-2536_cell_38.jpg", "PSEL-2538_cell_36_INDOOR.jpg", "PSEL-2537_cell_12_INDOOR.jpg", "PSEL-2538_cell_18.jpg", "RA17-0032_cell_17.jpg", "VCAD-0008_cell_54.jpg", "VCAD-0009_cell_37_INDOOR.jpg", "VCAD-0012_cell_30_INDOOR.jpg", "F1202-0018_cell_02.jpg", "VCAD-0009_cell_26_INDOOR.jpg", "RA02-0004_cell_27.jpg", "VCAD-0010_cell_14_INDOOR.jpg", "VCAD-0014_cell_55.jpg", "PSEL-2530_cell_23_INDOOR.jpg", "RA02-0020_cell_10.jpg", "F0202-0099_cell_32_INDOOR.jpg", "VCAD-0012_cell_29_INDOOR.jpg", "RA17-0032_cell_14.jpg", "PSEL-2536_cell_40.jpg", "RA02-0020_cell_02.jpg", "PSEL-2534_cell_04.jpg", "VCAD-0009_cell_09_INDOOR.jpg", "VCAD-0013_cell_14.jpg", "VCAD-0012_cell_02_INDOOR.jpg", "VCAD-0008_cell_19_INDOOR.jpg", "PSEL-2538_cell_01_INDOOR.jpg", "VCAD-0009_cell_23_INDOOR.jpg", "RA17-0032_cell_18.jpg", "PSEL-2534_cell_38.jpg", "PSEL-2538_cell_46_INDOOR.jpg", "VCAD-0013_cell_39_INDOOR.jpg", "VCAD-0014_cell_53.jpg", "F1202-0028_cell_15.jpg", "PSEL-2536_cell_17.jpg", "PSEL-2534_cell_03_INDOOR.jpg", "VCAD-0009_cell_36.jpg", "PSEL-2536_cell_53.jpg", "PSEL-2536_cell_33_INDOOR.jpg", "RA17-0032_cell_30.jpg", "VCAD-0014_cell_56_INDOOR.jpg", "PSEL-2536_cell_56_INDOOR.jpg", "PSEL-2534_cell_38_INDOOR.jpg", "VCAD-0009_cell_55_INDOOR.jpg", "VCAD-0004_cell_26.jpg", "PSEL-2536_cell_51.jpg", "PSEL-2534_cell_43.jpg", "PSEL-2537_cell_21.jpg", "PSEL-2537_cell_53_INDOOR.jpg", "PSEL-2538_cell_24_INDOOR.jpg", "PSEL-2536_cell_42.jpg", "VCAD-0012_cell_03.jpg", "VCAD-0008_cell_17_INDOOR.jpg", "PSEL-2534_cell_06_INDOOR.jpg", "VCAD-0014_cell_54.jpg", "PSEL-2534_cell_41.jpg", "VCAD-0009_cell_05.jpg", "PSEL-2536_cell_55.jpg", "RA17-0032_cell_31.jpg", "PSEL-2534_cell_04_INDOOR.jpg", "VCAD-0009_cell_11.jpg", "PSEL-2538_cell_44_INDOOR.jpg", "VCAD-0009_cell_27_INDOOR.jpg", "PSEL-2538_cell_20.jpg", "F0202-0112_cell_03_INDOOR.jpg", "VCAD-0009_cell_35_INDOOR.jpg", "PSEL-2537_cell_59_INDOOR.jpg", "F0202-0100_cell_18_INDOOR.jpg", "VCAD-0009_cell_54_INDOOR.jpg", "PSEL-2534_cell_41_INDOOR.jpg", "RA02-0036_cell_40.jpg", "PSEL-2536_cell_26_INDOOR.jpg", "PSEL-2537_cell_03_INDOOR.jpg", "VCAD-0008_cell_51_INDOOR.jpg", "PSEL-2534_cell_07.jpg", "PSEL-2536_cell_58.jpg", "F1501-0002_cell_47.jpg", "VCAD-0016_cell_45_INDOOR.jpg", "PSEL-2538_cell_04.jpg", "RA02-0020_cell_08.jpg", "VCAD-0009_cell_53.jpg", "VCAD-0008_cell_06.jpg", "VCAD-0008_cell_44_INDOOR.jpg", "PSEL-2538_cell_33.jpg", "PSEL-2538_cell_32.jpg", "VCAD-0009_cell_21_INDOOR.jpg", "VCAD-0013_cell_39.jpg", "PSEL-2534_cell_01_INDOOR.jpg", "PSEL-2538_cell_27.jpg", "VCAD-0004_cell_56_INDOOR.jpg", "PSEL-2537_cell_11_INDOOR.jpg", "RA02-0020_cell_15.jpg", "VCAD-0009_cell_22.jpg", "VCAD-0009_cell_03.jpg", "RA17-0032_cell_46.jpg", "PSEL-2537_cell_57_INDOOR.jpg", "VCAD-0009_cell_55.jpg", "VCAD-0009_cell_05_INDOOR.jpg", "VCAD-0008_cell_55.jpg", "PSEL-2530_cell_32_INDOOR.jpg", "VCAD-0009_cell_27.jpg", "PSEL-2531_cell_37_INDOOR.jpg", "F0202-0083_cell_21.jpg", "PSEL-2536_cell_54_INDOOR.jpg", "PSEL-2538_cell_52.jpg", "VCAD-0008_cell_02_INDOOR.jpg", "PSEL-2536_cell_36.jpg", "PSEL-2538_cell_06.jpg", "VCAD-0012_cell_28_INDOOR.jpg", "PSEL-2534_cell_13_INDOOR.jpg", "PSEL-2538_cell_18_INDOOR.jpg", "PSEL-2530_cell_18_INDOOR.jpg", "VCAD-0009_cell_51.jpg", "RA17-0032_cell_28.jpg", "VCAD-0009_cell_35.jpg", "VCAD-0013_cell_23.jpg", "VCAD-0016_cell_46_INDOOR.jpg", "PSEL-2537_cell_52_INDOOR.jpg", "VCAD-0008_cell_19.jpg", "VCAD-0013_cell_23_INDOOR.jpg", "F0202-0038_cell_06.jpg", "RA02-0020_cell_04.jpg", "PSEL-2536_cell_36_INDOOR.jpg", "RA02-0004_cell_25.jpg", "F0202-0041_cell_11.jpg", "VCAD-0009_cell_07.jpg", "VCAD-0008_cell_46.jpg", "VCAD-0013_cell_13_INDOOR.jpg", "VCAD-0010_cell_15_INDOOR.jpg", "PSEL-2538_cell_45_INDOOR.jpg", "PSEL-2538_cell_14_INDOOR.jpg", "PSEL-2536_cell_39_INDOOR.jpg", "F0202-0107_cell_33.jpg", "PSEL-2538_cell_25_INDOOR.jpg", "PSEL-2538_cell_12_INDOOR.jpg", "VCAD-0012_cell_05_INDOOR.jpg", "F0202-0033_cell_28.jpg", "VCAD-0006_cell_60.jpg", "RA02-0036_cell_44.jpg", "PSEL-2538_cell_32_INDOOR.jpg", "F0202-0099_cell_31_INDOOR.jpg", "PSEL-2538_cell_17_INDOOR.jpg", "VCAD-0008_cell_06_INDOOR.jpg", "VCAD-0008_cell_05.jpg", "PSEL-2538_cell_48_INDOOR.jpg", "VCAD-0004_cell_25_INDOOR.jpg", "PSEL-2538_cell_48.jpg", "VCAD-0012_cell_02.jpg", "PSEL-2536_cell_32_INDOOR.jpg", "VCAD-0009_cell_01.jpg", "PSEL-2538_cell_13_INDOOR.jpg", "PSEL-2538_cell_15_INDOOR.jpg", "F1501-0029_cell_50.jpg", "VCAD-0004_cell_25.jpg", "PSEL-2536_cell_43_INDOOR.jpg", "PSEL-2538_cell_09_INDOOR.jpg", "PSEL-2536_cell_32.jpg", "VCAD-0008_cell_53.jpg", "VCAD-0009_cell_02_INDOOR.jpg", "VCAD-0009_cell_04.jpg", "PSEL-2534_cell_08.jpg", "PSEL-2537_cell_30.jpg", "RA17-0032_cell_02.jpg", "RA02-0036_cell_07.jpg", "PSEL-2534_cell_46.jpg", "PSEL-2538_cell_34_INDOOR.jpg", "VCAD-0014_cell_57.jpg", "VCAD-0013_cell_13.jpg", "PSEL-2537_cell_01.jpg", "VCAD-0012_cell_05.jpg", "PSEL-2536_cell_33.jpg", "PSEL-2536_cell_17_INDOOR.jpg", "VCAD-0013_cell_37.jpg", "PSEL-2537_cell_20_INDOOR.jpg", "PSEL-2534_cell_05_INDOOR.jpg", "PSEL-2534_cell_10.jpg", "PSEL-2538_cell_34.jpg", "PSEL-2537_cell_58.jpg", "PSEL-2537_cell_03.jpg", "PSEL-2534_cell_02.jpg", "F0202-0112_cell_03.jpg", "PSEL-2538_cell_01.jpg", "VCAD-0009_cell_07_INDOOR.jpg", "PSEL-2538_cell_52_INDOOR.jpg", "VCAD-0014_cell_55_INDOOR.jpg", "F0202-0038_cell_20.jpg", "PSEL-2536_cell_56.jpg", "VCAD-0014_cell_53_INDOOR.jpg", "PSEL-2536_cell_40_INDOOR.jpg", "PSEL-2538_cell_35_INDOOR.jpg", "VCAD-0008_cell_45_INDOOR.jpg", "PSEL-2534_cell_37.jpg", "PSEL-2536_cell_31.jpg", "PSEL-2534_cell_11.jpg", "VCAD-0008_cell_51.jpg", "VCAD-0008_cell_53_INDOOR.jpg", "VCAD-0009_cell_51_INDOOR.jpg", "PSEL-2538_cell_30.jpg", "PSEL-2538_cell_44.jpg", "VCAD-0012_cell_31_INDOOR.jpg", "VCAD-0008_cell_01_INDOOR.jpg", "PSEL-2537_cell_30_INDOOR.jpg", "PSEL-2538_cell_16_INDOOR.jpg", "PSEL-2538_cell_06_INDOOR.jpg", "PSEL-2536_cell_31_INDOOR.jpg", "PSEL-2538_cell_02_INDOOR.jpg", "PSEL-2538_cell_36.jpg", "VCAD-0008_cell_56_INDOOR.jpg", "PSEL-2537_cell_17_INDOOR.jpg", "RA17-0032_cell_25.jpg", "VCAD-0008_cell_56.jpg", "PSEL-2537_cell_14_INDOOR.jpg", "VCAD-0008_cell_54_INDOOR.jpg", "VCAD-0009_cell_25.jpg", "VCAD-0008_cell_04.jpg", "PSEL-2534_cell_22.jpg", "PSEL-2530_cell_04_INDOOR.jpg", "RA02-0020_cell_13.jpg", "VCAD-0008_cell_45.jpg", "RA17-0032_cell_42.jpg", "VCAD-0008_cell_52_INDOOR.jpg", "VCAD-0013_cell_14_INDOOR.jpg", "VCAD-0016_cell_46.jpg", "PSEL-2534_cell_02_INDOOR.jpg", "PSEL-2534_cell_11_INDOOR.jpg", "RA17-0032_cell_21.jpg", "RA17-0032_cell_41.jpg", "PSEL-2534_cell_09.jpg", "VCAD-0010_cell_16_INDOOR.jpg", "VCAD-0004_cell_55_INDOOR.jpg", "PSEL-2538_cell_04_INDOOR.jpg", "VCAD-0013_cell_40.jpg", "PSEL-2538_cell_19_INDOOR.jpg", "PSEL-2538_cell_19.jpg", "PSEL-2536_cell_54.jpg", "F1202-0001_cell_16.jpg", "PSEL-2530_cell_45_INDOOR.jpg", "VCAD-0009_cell_11_INDOOR.jpg", "VCAD-0008_cell_44.jpg", "PSEL-2538_cell_03.jpg", "PSEL-2536_cell_45.jpg", "VCAD-0014_cell_56.jpg", "PSEL-2538_cell_05.jpg", "PSEL-2538_cell_03_INDOOR.jpg", "VCAD-0008_cell_18.jpg", "VCAD-0012_cell_31.jpg", "F1301-0013_cell_54.jpg", "PSEL-2536_cell_53_INDOOR.jpg", "VCAD-0010_cell_16.jpg", "VCAD-0014_cell_58.jpg", "PSEL-2537_cell_57.jpg", "RA02-0020_cell_18.jpg", "VCAD-0013_cell_38.jpg", "RA17-0032_cell_19.jpg", "VCAD-0008_cell_18_INDOOR.jpg", "VCAD-0008_cell_01.jpg", "VCAD-0008_cell_55_INDOOR.jpg", "PSEL-2536_cell_55_INDOOR.jpg", "PSEL-2529_cell_65_INDOOR.jpg", "PSEL-2534_cell_48.jpg", "F0202-0050_cell_33.jpg", "F0202-0090_cell_20.jpg", "PSEL-2537_cell_59.jpg", "VCAD-0009_cell_12.jpg", "VCAD-0009_cell_24.jpg", "VCAD-0013_cell_35.jpg", "PSEL-2537_cell_16_INDOOR.jpg", "PSEL-2536_cell_45_INDOOR.jpg", "PSEL-2534_cell_24.jpg", "VCAD-0013_cell_36.jpg", "VCAD-0014_cell_54_INDOOR.jpg", "PSEL-2534_cell_40.jpg", "PSEL-2530_cell_41_INDOOR.jpg", "F0202-0108_cell_30.jpg", "RA17-0032_cell_29.jpg", "RA17-0032_cell_07.jpg", "RA02-0004_cell_23.jpg", "F0202-0090_cell_20_INDOOR.jpg", "RA17-0032_cell_12.jpg", "PSEL-2534_cell_37_INDOOR.jpg", "VCAD-0010_cell_15.jpg", "VCAD-0008_cell_02.jpg", "PSEL-2536_cell_16_INDOOR.jpg", "PSEL-2538_cell_25.jpg", "VCAD-0010_cell_13_INDOOR.jpg", "RA02-0004_cell_29.jpg", "PSEL-2534_cell_42.jpg", "VCAD-0009_cell_02.jpg", "RA17-0032_cell_08.jpg", "RA17-0032_cell_33.jpg", "VCAD-0013_cell_37_INDOOR.jpg", "VCAD-0013_cell_38_INDOOR.jpg", "VCAD-0009_cell_03_INDOOR.jpg", "VCAD-0008_cell_17.jpg", "VCAD-0012_cell_29.jpg", "PSEL-2536_cell_42_INDOOR.jpg", "PSEL-2534_cell_07_INDOOR.jpg", "PSEL-2536_cell_57.jpg", "PSEL-2534_cell_39_INDOOR.jpg", "VCAD-0008_cell_52.jpg", "VCAD-0009_cell_24_INDOOR.jpg", "PSEL-2536_cell_51_INDOOR.jpg", "VCAD-0009_cell_08.jpg", "VCAD-0012_cell_03_INDOOR.jpg", "PSEL-2538_cell_47_INDOOR.jpg", "VCAD-0004_cell_55.jpg", "PSEL-2538_cell_02.jpg", "VCAD-0016_cell_45.jpg", "VCAD-0004_cell_26_INDOOR.jpg", "F0202-0116_cell_35.jpg", "VCAD-0016_cell_47.jpg", "VCAD-0009_cell_09.jpg", "PSEL-2537_cell_01_INDOOR.jpg", "PSEL-2534_cell_21.jpg", "PSEL-2534_cell_39.jpg", "VCAD-0009_cell_26.jpg", "PSEL-2536_cell_38_INDOOR.jpg", "RA02-0036_cell_12.jpg", "PSEL-2534_cell_03.jpg", "VCAD-0009_cell_25_INDOOR.jpg", "VCAD-0009_cell_23.jpg", "PSEL-2538_cell_45.jpg", "VCAD-0009_cell_21.jpg", "PSEL-2538_cell_46.jpg", "PSEL-2534_cell_23.jpg", "PSEL-2536_cell_59_INDOOR.jpg", "RA02-0020_cell_12.jpg", "F0202-0083_cell_21_INDOOR.jpg", "VCAD-0012_cell_04_INDOOR.jpg", "PSEL-2534_cell_01.jpg", "VCAD-0013_cell_36_INDOOR.jpg", "PSEL-2537_cell_10.jpg", "PSEL-2537_cell_21_INDOOR.jpg", "PSEL-2538_cell_20_INDOOR.jpg", "VCAD-0008_cell_16.jpg", "VCAD-0009_cell_04_INDOOR.jpg", "VCAD-0008_cell_16_INDOOR.jpg", "VCAD-0013_cell_15_INDOOR.jpg", "PSEL-2538_cell_05_INDOOR.jpg", "VCAD-0014_cell_58_INDOOR.jpg", "PSEL-2536_cell_41_INDOOR.jpg", "PSEL-2534_cell_12_INDOOR.jpg", "VCAD-0009_cell_36_INDOOR.jpg", "VCAD-0008_cell_03.jpg", "VCAD-0009_cell_22_INDOOR.jpg", "PSEL-2534_cell_42_INDOOR.jpg", "VCAD-0009_cell_12_INDOOR.jpg", "VCAD-0012_cell_04.jpg", "VCAD-0010_cell_14.jpg", "VCAD-0009_cell_53_INDOOR.jpg", "VCAD-0008_cell_04_INDOOR.jpg", "VCAD-0016_cell_34.jpg", "PSEL-2534_cell_12.jpg", "RA17-0032_cell_06.jpg", "VCAD-0010_cell_13.jpg", "PSEL-2534_cell_09_INDOOR.jpg", "PSEL-2536_cell_50_INDOOR.jpg", "RA02-0020_cell_07.jpg", "RA02-0036_cell_10.jpg", "VCAD-0013_cell_35_INDOOR.jpg", "RA17-0032_cell_04.jpg", "VCAD-0009_cell_10_INDOOR.jpg", "PSEL-2537_cell_20.jpg", "VCAD-0008_cell_57_INDOOR.jpg", "VCAD-0012_cell_28.jpg", "PSEL-2536_cell_50.jpg", "PSEL-2530_cell_40_INDOOR.jpg", "VCAD-0006_cell_60_INDOOR.jpg", "RA02-0036_cell_08.jpg", "PSEL-2537_cell_11.jpg", "VCAD-0014_cell_57_INDOOR.jpg", "PSEL-2538_cell_47.jpg", "VCAD-0008_cell_03_INDOOR.jpg", "PSEL-2534_cell_40_INDOOR.jpg", "VCAD-0012_cell_06.jpg", "PSEL-2537_cell_58_INDOOR.jpg", "PSEL-2534_cell_47.jpg", "VCAD-0009_cell_54.jpg", "PSEL-2534_cell_13.jpg", "PSEL-2538_cell_35.jpg", "PSEL-2536_cell_57_INDOOR.jpg", "VCAD-0013_cell_40_INDOOR.jpg", "PSEL-2534_cell_08_INDOOR.jpg", "VCAD-0009_cell_08_INDOOR.jpg", "VCAD-0009_cell_01_INDOOR.jpg", "VCAD-0008_cell_46_INDOOR.jpg", "PSEL-2538_cell_30_INDOOR.jpg", "PSEL-2536_cell_59.jpg", "VCAD-0013_cell_15.jpg", "F1202-0009_cell_42.jpg"]


# working all below

missing_images = []
broken_annotations = []
broken_annotation_image_ids = []
suggested_skip_list = []


# set by Dawn, please note this as this is how it is doing the UVF annotation corrects
uvf_defect_categories = {
    'square':   1,
    'ring':     2,
    'crack':    3,
    'bright_crack': 4,
    'hotspot': 5,
    'finger_corrosion': 6,
    'near_busbar': 7,
    'busbar_crack': 8,
    'shattered': 9,
    'misc': 10
}


# categories matching the uvf_detect_category id's 
coco_categories = [
    {
      "id": 1,
      "name": "square",
      "supercategory": ""
    },
    {
          "id": 2,
          "name": "ring",
          "supercategory": ""
    },
    {
          "id": 3,
          "name": "crack",
          "supercategory": ""
    },
    {
          "id": 4,
          "name": "bright_crack",
          "supercategory": ""
    },
    {
          "id": 5,
          "name": "hotspot",
          "supercategory": ""
    },
    {
          "id": 6,
          "name": "finger_corrosion",
          "supercategory": ""
    },
    {
          "id": 7,
          "name": "near_busbar",
          "supercategory": ""
    },
    {
          "id": 8,
          "name": "busbar_crack",
          "supercategory": ""
    },
    {
          "id": 9,
          "name": "shattered",
          "supercategory": ""
    },
    {
          "id": 10,
          "name": "misc",
          "supercategory": ""
    }
]

# coco lisences 
coco_licenses  = [
    {
      "name": "",
      "id": 0,
      "url": ""
    }
  ]

# coco info parts
coco_info = {
    "contributor": "",
    "date_created": "",
    "description": "",
    "url": "",
    "version": "",
    "year": ""
  }


global_annotation_id = -1
#print(global_annotation_id)
def create_image_json(filename, id):
    image_path =  os.path.join(images_dir, filename)
    try:
        image = Image.open(image_path)
        width, height = image.size

        image_json = {
          "id": id,
          "width": width,
          "height": height,
          "file_name": filename,
          "license": 0,
          "flickr_url": "",
          "coco_url": "",
          "date_captured": 0
        }
        
    except Exception:
        #print(f"the image {filename} did not exist or had an error")
        missing_images.append(f"{filename}")
        suggested_skip_list.append(f"{filename}")
        image_json = {
          "id": id,
          "width": None,
          "height": None,
          "file_name": filename,
          "license": 0,
          "flickr_url": "Image was Lost or Not found please find",
          "coco_url": "",
          "date_captured": 0
        }

    return image_json
    
    
def create_annotations(regions, region_attributes, image_id):
    
    # set base systems
    annotations = []
    global global_annotation_id
    for reg, attribute, in zip (regions, region_attributes):
        global_annotation_id = global_annotation_id+1
        #print(global_annotation_id)
        try:
            #turn into normal jsons
            reg = parse_region_attributes(reg)
            attribute = parse_region_attributes(attribute)
            feature =attribute['feature']
            
            # get category index for color purposing
            category_index = uvf_defect_categories.get(feature)
            # this will typically catch spelling or annotation issues
            if(category_index is None):
                print(f"{feature}_")
                
                print(f"we broke on this one\n{image_id}\n{reg}\n{attribute}")
                category_index = "broken annotation"
                segmentation = None
                area = None
                bbox = None
                broken_annotations.append({
                    'image_id': image_id,
                    'region_shape_attributes': reg,
                    'region_attributes': attribute
                    })
                broken_annotation_image_ids.append(image_id)


                #exit()
            else:
           
                # need to create the versions for the different type of polygon systems
                #skips for non polygons/polylines, see bottom of code for potential solutions for all annotation region types
                
                if (reg["name"] == "polygon") or (reg["name"] == "polyline"):
                    # Generate the list of x, y points
                    #if(reg["all_points_x"].size() <3 or reg["all_points_y"].len <3):
                        
                    points = list(zip(reg["all_points_x"], reg["all_points_y"]))
                    #print(reg["all_points_x"])
                    
                    



                    # Convert to flat list of floats for COCO segmentation
                    segmentation = [list(map(float, sum(points, ())))]
                    #print(len(segmentation[0]))
                    #print("here")
                    if len(segmentation[0]) <=4:
                        #print(len(segmentation[0]))
                        print("number of points error")
                        print(f"we broke on this one\n{image_id}\n{reg}\n{attribute}")
                        category_index = "broken annotation"
                        segmentation = None
                        area = None
                        bbox = None
                        broken_annotations.append({
                            'image_id': image_id,
                            'region_shape_attributes': reg,
                            'region_attributes': attribute
                            })
                        broken_annotation_image_ids.append(image_id)
                        #exit()
                    else:
                            
                        # Convert to NumPy array of shape (n, 1, 2) for OpenCV ops
                        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = [x, y, w, h]

                        area = float(cv2.contourArea(contour))

                # elif (reg["name"] == "rect"):
                #     x, y, w, h = reg["x"], reg["y"], reg["width"], reg["height"]
                #     segmentation = [[
                #         float(x), float(y),
                #         float(x + w), float(y),
                #         float(x + w), float(y + h),
                #         float(x), float(y + h)
                #     ]]
                #     bbox = [x, y, w, h]
                #     area = float(w * h)
                else:         
                    
                    #print(f"error on this annotation ")
                    #annotation_id = "broken_annotation"
                    # if (reg["name"] == "ellipse" or reg["name"] == "circle"):
                    #     "skipped"
                    # else:
                        category_index = "broken annotation"
                        segmentation = None
                        area = None
                        bbox = None
                        broken_annotations.append({
                            'image_id': image_id,
                            'region_shape_attributes': reg,
                            'region_attributes': attribute
                            })
                        broken_annotation_image_ids.append(image_id)
                


            
        except Exception:
            
            # sets the data to not exist and then I can pull it later
            category_index = "broken annotation"
            segmentation = None
            area = None
            bbox = None
            broken_annotations.append({
                    'image_id': image_id,
                    'region_shape_attributes': reg,
                    'region_attributes': attribute
                    })
            broken_annotation_image_ids.append(image_id)
        
        # build the annotation.json and then append it to the list
        annotation_json = {
            "id": global_annotation_id,
            "image_id": image_id,
            "category_id": category_index,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
            "attributes": {
            "occluded": "false"
            }
        }
        
        annotations.append(annotation_json)
        
    return annotations


def parse_region_attributes(region_str: str) -> dict:
    """Parse region attributes from JSON string."""
    return json.loads(region_str)



def create_dataframe(csv_path):
    df = pd.read_csv(csv_path, header=0)
    return df

def process_annotations(csv_path: str, images_dir: str, output_dir: str):
   
    # read the csv into a dataframe and then group it so that it is easier to work with
    data_df = create_dataframe(csv_path)
    image_data_frames = [group for _, group in data_df.groupby("filename")]

    
    # stores the jsons and dicts that gets adjusted into the full coco.json
    image_dictionary = {}
    image_jsons = []    # holds the coco jsons for the image identifications
    coco_annotations = []

    image_id = 0;       # this is needing to be pulled from the main source you take the measurements  -- this isnt the DARTS image id, but rather the coco pairing id
    
    for image_file_df in image_data_frames:
        
        # get the file name
        filename = image_file_df.iloc[0, 0]

        # check to see if we should skip it or not
        if filename in custom_suggested_skip_list:
            continue

        # this means that the image existed so likes go ahead and build it
        image_id = image_id +1
        image_dictionary[f"{image_id}"] = filename

        
        # get the attributes and datas
        shape_attributes = image_file_df["region_shape_attributes"].to_list()
        region_attributes = image_file_df["region_attributes"].to_list()
        
        # create the header image json
        image_json = create_image_json(filename, image_id)
        image_jsons.append(image_json)        

        # craete the annottions and then add them to the bulk annotation list
        image_annotations = create_annotations(shape_attributes, region_attributes, image_id)

        for annotation in image_annotations:
            coco_annotations.append(annotation)

    # now create the full json for the coco annotation conversion
    coco_data = {
        "licenses":     coco_licenses,
        "info":         coco_info,
        "categories":   coco_categories,
        "images":       image_jsons,
        "annotations":  coco_annotations
        
    }

    # now this is for the debugging and seeing where things went wrong
    unique_list = list(set(broken_annotation_image_ids)) # make it a unique list
    broken_annotation_image_list = []
    
    # pair the ids back up with the images
    for broken_image_id in unique_list:
        broken_annotation_image_list.append(image_dictionary[f"{broken_image_id}"])
        suggested_skip_list.append(image_dictionary[f"{broken_image_id}"])

    # remove duplicates from the suggested skip list that might have both bad annotations and missing image.png
    unique_skip_list = list(set(suggested_skip_list))


    # runs all the outputs of the the system into the desired output_directory
    
    # attempt to export main it to cocoformat with broken ones
    try:
        with open(os.path.join(output_dir,"coco_conversion_output.json"), "w") as json_file:
            json.dump(coco_data, json_file, indent=4) # indent for pretty-printing
        print("Data successfully written to coco_conversion_output.json")
    except IOError as e:
        print(f"Error writing to file: {e}")

    # attempt to print out the missing images
    try:
        with open(os.path.join(output_dir, "coco_conversion_mising_images.txt"), "w") as missing_images_file:
            for missing_image in missing_images:
                missing_images_file.write(f"{missing_image}\n")
        print("Data successfully written to coco_conversion_mising_images.txt")
    except IOError as e:
        print(f"Error writing to file: {e}")

    # attempt to print the non converted annotations
    try:
        with open(os.path.join(output_dir,"coco_conversion_broken_annotations.json"), "w") as json_file:
            json.dump(broken_annotations, json_file, indent=4) # indent for pretty-printing
        print("Data successfully written to coco_conversion_broken_annotations.json")
    except IOError as e:
        print(f"Error writing to file: {e}")

    # print out broken annotation list
    try:
        with open(os.path.join(output_dir,"coco_conversion_broken_annotation_images.txt"), "w") as broken_annotation_image_list_file:  # I am so so sorry for the other cs majors to hate on it
            for broken_image in broken_annotation_image_list:
                broken_annotation_image_list_file.write(f"{broken_image}\n")
        print("Data successfully written to coco_conversion_broken_annotation_images.txt")
    except IOError as e:
        print(f"Error writing to file: {e}")

    # print out suggested skiplist 
    try:
        with open(os.path.join(output_dir,"coco_conversion_suggested_skip.txt"), "w") as unique_skip_list_file:  # I am so so sorry for the other cs majors to hate on it
            for broken_image in unique_skip_list:
                unique_skip_list_file.write(f"{broken_image}\n")
        print("Data successfully written to coco_conversion_suggested_skip.txt")
    except IOError as e:
        print(f"Error writing to file: {e}")

def main():
    process_annotations(csv_path, images_dir, output_dir)

if __name__ == "__main__":
    main()

