import os
import json
import shutil

base_dir = r"/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/Annotated"
images_dir = os.path.join(base_dir, "Images")
split_dir = os.path.join(base_dir, "split_data")

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

copy_images("train.json", "train")
copy_images("test.json", "test")
