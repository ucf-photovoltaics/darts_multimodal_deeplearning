import os
import json

# Update with your actual path
json_path = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/Annotated/split_data/train.json"
image_dir = "/home/axo360/CSE_MSE_RXF131/cradle-members/sdle/axo360/UVF_Script_Test/Annotated/split_data/train"

with open(json_path, 'r') as f:
    coco = json.load(f)

images = coco.get("images", [])
annotations = coco.get("annotations", [])

image_filenames = set(os.listdir(image_dir))
image_id_to_filename = {img["id"]: img["file_name"] for img in images}

used_image_ids = set()
valid_anno_count = 0
missing_images = 0
empty_annos = 0

for anno in annotations:
    img_id = anno["image_id"]
    filename = image_id_to_filename.get(img_id)

    if not filename or filename not in image_filenames:
        print(f"[MISSING] {filename} not found for image_id {img_id}")
        missing_images += 1
        continue

    if not anno.get("segmentation") or not anno.get("bbox"):
        print(f"[EMPTY] Annotation {anno['id']} has no segmentation or bbox")
        empty_annos += 1
        continue

    used_image_ids.add(img_id)
    valid_anno_count += 1

print("\n=== Summary ===")
print(f"Total images in COCO: {len(images)}")
print(f"Images in folder: {len(image_filenames)}")
print(f"Images used in annotations: {len(used_image_ids)}")
print(f"Valid annotations: {valid_anno_count}")
print(f"Missing images: {missing_images}")
print(f"Empty annotations: {empty_annos}")
