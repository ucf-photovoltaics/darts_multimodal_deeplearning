import os
import sys
import pandas as pd
from torchvision.io import read_image
from glob import glob
import cv2

dataset_dir='/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/sxl2318/classification/dataset/UVF/val'
csv_read_path = dataset_dir + '/all_annotations.csv'

def load_images_from_folder(folder):

    img_list = []
    files = os.listdir (folder)
    for name in files:
        if(name.endswith("jpg")):
           img_list.append(name)
    
    return img_list

def main():
    img_labels_rows = pd.read_csv(csv_read_path)
    img_labels_rows = img_labels_rows.dropna() #dropping one with NAN
    img_labels_rows = img_labels_rows[img_labels_rows.region_attributes != '{}']
    img_labels_rows = img_labels_rows[img_labels_rows.region_attributes != '{"feature":"","number":""}']

    images = load_images_from_folder(dataset_dir)
    index = []

    for i, row in enumerate (img_labels_rows['filename']):
        if row in images:
            index.append (i)

    df = img_labels_rows.iloc[index]
    breakpoint()
    df.to_csv(os.path.join (dataset_dir,'annotation_from_script.csv'))


if __name__ == "__main__":
  main()
  print ('All done')