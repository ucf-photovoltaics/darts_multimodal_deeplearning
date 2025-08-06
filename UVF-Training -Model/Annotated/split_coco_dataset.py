# Author: Andres Ojeda Sainz
# Date: 06/24/2025
# Script to split COCO-formatted dataset into 80% training and 20% testing.
# adjusted  by Dawn Balaschak on 07/17/2025 with a few pathing adjustments for cleaned up annotations
# and to include moving the images aswell

import json
import os
import random
import shutil


# === Paths ===
# adjusted by Dawn with new annotations 
base_dir =       r"/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/Annotated"
output_dir =     r"/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/Annotated/split_data"
images_dir =     r"/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/Annotated/Images"
full_coco_path = r"/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/CocoConversionTools/Outputs/coco_conversion_output.json"


def copy_images(json_file, split_name):
    with open(os.path.join(split_dir, json_file), "r") as f:
        data = json.load(f)

    dest_dir = os.path.join(split_dir, split_name)
    os.makedirs(dest_dir, exist_ok=True)

    copied = 0
    for img in data["images"]:
        src_path = os.path.join(images_dir, img["file_name"])
        dest_path = os.path.join(dest_dir, img["file_name"])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            copied += 1
        else:
            print(f"[WARNING] Missing file: {img['file_name']}")

    print(f"✅ Copied {copied} images to {split_name}/")


split_dir = output_dir
os.makedirs(output_dir, exist_ok=True)

# === Load full COCO ===
with open(full_coco_path, "r") as f:
    coco_data = json.load(f)

# === Shuffle and split ===
images = coco_data["images"]
annotations = coco_data["annotations"]
categories = coco_data["categories"]

random.shuffle(images)
split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
test_images = images[split_idx:]

# === Build image_id lookup ===
train_ids = {img["id"] for img in train_images}
test_ids = {img["id"] for img in test_images}

train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
test_annotations = [ann for ann in annotations if ann["image_id"] in test_ids]

# === Save new COCOs ===
train_json = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}
test_json = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": categories
}

with open(os.path.join(output_dir, "train.json"), "w") as f:
    json.dump(train_json, f, indent=4)

with open(os.path.join(output_dir, "test.json"), "w") as f:
    json.dump(test_json, f, indent=4)

print("New train and test splits jsons created.")
print(f"Saved to: {output_dir}")
print(f"Train images: {len(train_images)} | Test images: {len(test_images)}")
print("Now copying over the images into the respective folders")

copy_images("train.json", "train")
copy_images("test.json", "test")

print("Successfully finishing splitting the cocodataset into 80% train, 20% testing")