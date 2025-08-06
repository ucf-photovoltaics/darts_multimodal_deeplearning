# create to check and make sure the segmentations are correct

import json
import os

def is_valid_segmentation(segmentation, iscrowd):
    if iscrowd == 1:
        return isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation
    if not isinstance(segmentation, list):
        return False
    if len(segmentation) == 0:
        return False
    # Must be a list of flat polygons
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            return False
        if any(not isinstance(coord, (int, float)) for coord in poly):
            return False
    return True

def check_coco_json(json_path):
    with open(json_path, "r") as f:
        coco_data = json.load(f)

    total = len(coco_data["annotations"])
    invalid = 0
    invalid_ids = []

    for ann in coco_data["annotations"]:
        segm = ann.get("segmentation", [])
        iscrowd = ann.get("iscrowd", 0)
        if not is_valid_segmentation(segm, iscrowd):
            invalid += 1
            invalid_ids.append(ann.get("id", "unknown"))

    print(f"Total annotations checked: {total}")
    print(f"Invalid segmentations: {invalid}")
    if invalid:
        print("Invalid annotation IDs (or unknown):")
        for i in invalid_ids[:20]:
            print(f" - {i}")
        if len(invalid_ids) > 20:
            print("... (truncated)")

if __name__ == "__main__":
    # Replace this path with your full JSON path thing
    json_file = "/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/Dawn - UVF/UVF_Script_Test/CocoConversionTools/Outputs/coco_conversion_output.json"
    if not os.path.exists(json_file):
        print(f"File not found: {json_file}")
    else:
        check_coco_json(json_file)
