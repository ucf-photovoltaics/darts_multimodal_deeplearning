import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def parse_region_attributes(region_str: str) -> dict:
    """Parse region attributes from JSON string."""
    try:
        return json.loads(region_str)
    except json.JSONDecodeError:
        return {}

def create_mask_from_rectangles(rectangles: List[dict], width: int, height: int) -> np.ndarray:
    """Create mask from rectangle annotations."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for rect in rectangles:
        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    return mask

def create_mask_from_polygons(polygons: List[dict], width: int, height: int) -> np.ndarray:
    """Create mask from polygon annotations."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        points = np.array(list(zip(poly['all_points_x'], poly['all_points_y'])))
        cv2.fillPoly(mask, [points], 255)
    return mask

def create_colored_mask(defect_class: str, width: int, height: int) -> np.ndarray:
    """Create colored mask based on defect class."""
    print(f"Processing defect class: {defect_class}")
    
    # Define colors explicitly for each defect type
    color_map = {
        'Contact_FrontGridInterruption': (0.215687, 0.494118, 0.721569),     # Blue
        'Contact_NearSolderPad': (0.215687, 0.494118, 0.721569),             # Blue
        'Contact_BeltMarks': (1.0, 0.4980392156862745, 0.0),                 # Orange
        'Contact_Corrosion': (1.0, 0.4980392156862745, 0.0),                 # Orange
        'Interconnect_Disconnected': (0.596078, 0.305882, 0.639216),         # Purple
        'Interconnect_HighlyResistive': (0.596078, 0.305882, 0.639216),      # Purple
        'Interconnect_BrightSpot': (0.596078, 0.305882, 0.639216),           # Purple
        'Crack_Closed': (0.894118, 0.101961, 0.109804),                      # Red
        'Crack_Resistive': (0.894118, 0.101961, 0.109804),                   # Red
        'Crack_Isolated': (0.894118, 0.101961, 0.109804)                     # Red
    }
    
    # Get the color for this defect class
    color = color_map.get(defect_class, (0.0, 0.0, 0.0))  # Default to black if not found
    
    # Create a 3-channel mask with the specific color
    mask = np.zeros((height, width, 3), dtype=np.float32)
    
    # Fill each channel with the corresponding color component
    mask[:,:,0] = color[0]  # Blue channel
    mask[:,:,1] = color[1]  # Green channel
    mask[:,:,2] = color[2]  # Red channel
    
    # Verify the mask values
    print(f"Color values: {color}")
    print(f"Mask min/max values: {np.min(mask)}, {np.max(mask)}")
    print(f"Mask shape: {mask.shape}")
    
    return mask

def process_annotations(csv_path: str, images_dir: str, output_dir: str):
    """Process annotations and create masks with colored overlays."""
    # Create output directories
    masks_dir = os.path.join(output_dir, 'masks')
    overlays_dir = os.path.join(output_dir, 'overlays')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    
    # Read CSV file using csv module
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Group annotations by filename
        image_annotations = {}
        for row in reader:
            filename = row[header.index('filename')]
            if filename not in image_annotations:
                image_annotations[filename] = []
            
            # Get region attributes
            region_idx = header.index('region_shape_attributes')
            shape_attrs_str = row[region_idx]
            defect_class_idx = header.index('region_attributes')
            defect_class_str = row[defect_class_idx]
            
            image_annotations[filename].append({
                'shape_attrs': shape_attrs_str,
                'defect_class': defect_class_str
            })
    
    # Process each image and its annotations
    for filename, annotations in image_annotations.items():
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Load original image
        image = cv2.imread(img_path)
        height, width = image.shape[:2]
        
        # Initialize combined binary mask and colored mask
        combined_binary_mask = np.zeros((height, width), dtype=np.uint8)
        combined_colored_mask = np.zeros((height, width, 3), dtype=np.float32)
        
        # Process each annotation for this image
        for annotation in annotations:
            try:
                # Parse shape attributes
                shape_attrs = parse_region_attributes(annotation['shape_attrs'])
                
                # Parse defect class
                try:
                    if isinstance(annotation['defect_class'], int):
                        defect_class = str(annotation['defect_class'])
                    else:
                        defect_class_dict = json.loads(annotation['defect_class'])
                        defect_class = defect_class_dict.get('Defect_Class', '')
                except json.JSONDecodeError:
                    print(f"Error parsing defect class for {filename}")
                    continue
                
                # Create binary mask based on shape type
                if shape_attrs.get('name') == 'rect':
                    binary_mask = create_mask_from_rectangles([shape_attrs], width, height)
                else:
                    binary_mask = create_mask_from_polygons([shape_attrs], width, height)
                
                # Update combined binary mask
                combined_binary_mask = np.maximum(combined_binary_mask, binary_mask)
                
                # Create colored mask for this defect
                colored_mask = create_colored_mask(defect_class, width, height)
                
                # Apply color only where this defect's binary mask is active
                mask_indices = binary_mask == 255
                combined_colored_mask[mask_indices] = colored_mask[mask_indices]
                
            except (KeyError, IndexError) as e:
                print(f"Error processing annotation for {filename}: {str(e)}")
                continue
        
        # Save binary mask
        mask_path = os.path.join(masks_dir, f"{Path(filename).stem}_mask.png")
        cv2.imwrite(mask_path, combined_binary_mask)
        
        # Create and save overlay
        overlay = image.astype(np.float32) / 255.0
        mask_indices = combined_binary_mask == 255
        overlay[mask_indices] = overlay[mask_indices] * 0.5 + combined_colored_mask[mask_indices] * 0.5
        overlay_path = os.path.join(overlays_dir, f"{Path(filename).stem}_overlay.png")
        cv2.imwrite(overlay_path, (overlay * 255.0).astype(np.uint8))

def main():
    """Example usage."""
    csv_path = "/home/josephr/Desktop/UCFPhotovoltatics/MultiModal/darts_multimodal_deeplearning/UCF-EL-Defect-Annotations/Test/via_project_21May2025_22h39m_csv(1).csv"
    images_dir = "/home/josephr/Desktop/UCFPhotovoltatics/MultiModal/darts_multimodal_deeplearning/UCF-EL-Defect-Annotations/Test/InputImages"
    output_dir = "/home/josephr/Desktop/UCFPhotovoltatics/MultiModal/darts_multimodal_deeplearning/UCF-EL-Defect-Annotations/Test/OutputMasks"
    process_annotations(csv_path, images_dir, output_dir)

if __name__ == "__main__":
    main()