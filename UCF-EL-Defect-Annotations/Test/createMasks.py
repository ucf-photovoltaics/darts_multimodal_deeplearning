import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Define defect categories and their colors
defect_categories = {
    'Contact_FrontGridInterruption': 0,
    'Contact_NearSolderPad': 0,
    'Contact_BeltMarks': 1,
    'Contact_Corrosion': 1,
    'Interconnect_Disconnected': 2,
    'Interconnect_HighlyResistive': 2,
    'Interconnect_BrightSpot': 2,
    'Crack_Closed': 3,
    'Crack_Resistive': 3,
    'Crack_Isolated': 3
}

# Color map for visualization
cmaplist = [
    (0.001462, 0.000466, 0.013866, 1.0),     # Blue group
    (0.8941176470588236, 0.10196078431372549, 0.10980392156862745, 1.0),  # Orange group
    (0.21568627450980393, 0.49411764705882355, 0.7215686274509804, 1.0),   # Purple group
    (0.596078431372549, 0.3058823529411765, 0.6392156862745098, 1.0)       # Red group
]

def parse_region_attributes(region_str: str) -> dict:
    """Parse region attributes from JSON string."""
    return json.loads(region_str)

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
    category_index = defect_categories.get(defect_class, 0)
    color = cmaplist[category_index][:3]  # Remove alpha channel
    return np.full((height, width, 3), color, dtype=np.float32)

def process_annotations(csv_path: str, images_dir: str, output_dir: str):
    """
    Process annotations and create masks with colored overlays.
    
    Args:
        csv_path: Path to CSV file containing annotations
        images_dir: Directory containing original images
        output_dir: Output directory for masks and overlays
    """
    # Create output directories
    masks_dir = os.path.join(output_dir, 'masks')
    overlays_dir = os.path.join(output_dir, 'overlays')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    
    # Read CSV file
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        
        for line in f:
            values = line.strip().split(',')
            
            filename = values[header.index('filename')]
            img_path = os.path.join(images_dir, filename)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Load original image
            image = cv2.imread(img_path)
            height, width = image.shape[:2]
            
            # Initialize combined mask
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Process each region
            for i in range(int(values[header.index('region_count')])):
                region_idx = header.index('region_shape_attributes') + i * len(header)
                
                shape_attrs = json.loads(values[region_idx])
                defect_class = json.loads(values[header.index('region_attributes') + i * len(header)])['Defect_Class']
                
                if shape_attrs['name'] == 'rect':
                    mask = create_mask_from_rectangles([shape_attrs], width, height)
                else:
                    mask = create_mask_from_polygons([shape_attrs], width, height)
                
                # Apply defect-specific color
                colored_mask = create_colored_mask(defect_class, width, height)
                colored_mask[mask == 255] = colored_mask[mask == 255] * 0.5
                
                # Combine with existing mask
                combined_mask = np.maximum(combined_mask, mask)
            
            # Save binary mask
            mask_path = os.path.join(masks_dir, f"{Path(filename).stem}_mask.png")
            cv2.imwrite(mask_path, combined_mask)
            
            # Create and save overlay
            overlay = image.copy().astype(np.float32) / 255.0
            overlay[combined_mask == 255] = overlay[combined_mask == 255] * 0.5
            
            overlay_path = os.path.join(overlays_dir, f"{Path(filename).stem}_overlay.png")
            cv2.imwrite(overlay_path, overlay * 255.0)

def main():
    """Example usage."""
    csv_path = "path/to/annotations.csv"
    images_dir = "path/to/images"
    output_dir = "path/to/output"
    
    process_annotations(csv_path, images_dir, output_dir)

if __name__ == "__main__":
    main()