import os
import json
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors

# Loads and parses each annotation from the CSV file.
def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    
    # Parse shape attributes and region attributes
    df['points'] = df['region_shape_attributes'].apply(json.loads)
    df['attributes'] = df['region_attributes'].apply(json.loads)
    
    return df

# Creates a set of tuples to represent each type of defect and it's color mapping.
def create_color_map():
    return {
        'Contact_FrontGridInterruption': (0.215687, 0.494118, 0.721569, 1.0),     # Blue
        'Contact_NearSolderPad': (0.215687, 0.494118, 0.721569, 1.0),             # Blue
        'Contact_BeltMarks': (1.0, 0.5, 0.0, 1.0),                                # Orange
        'Contact_Corrosion': (1.0, 0.5, 0.0, 1.0),                                # Orange
        'Interconnect_Disconnected': (0.596078, 0.305882, 0.639216),              # Purple
        'Interconnect_HighlyResistive': (0.596078, 0.305882, 0.639216),           # Purple
        'Interconnect_BrightSpot': (0.596078, 0.305882, 0.639216),                # Purple
        'Crack_Closed': (0.894118, 0.101961, 0.109804),                           # Red
        'Crack_Resistive': (0.894118, 0.101961, 0.109804),                        # Red
        'Crack_Isolated': (0.894118, 0.101961, 0.109804)                          # Red
    }

# Applies a colored mask to the input image. Takes in a preloaded image, the points that make up the mask
# (numpy array), color of the type of defect (RGBA), and the alpha transparency (0.3). 
def apply_mask(image, points, color, alpha=0.3):
    mask = np.zeros_like(image, dtype=np.uint8)
    points = np.array(points)
    
    # Draw filled polygon on mask
    cv2.fillPoly(mask, [points], color)
    
    # Apply transparency
    result = image.copy()
    mask = mask.astype(float) * alpha
    result[mask.sum(axis=2) > 0] = result[mask.sum(axis=2) > 0] * (1 - alpha) + mask[mask.sum(axis=2) > 0]
    
    return result

# Processes each individual image and the annotations associated with it. Takes in the image_path (string), 
# the annotation(s) (df), and the color map (tuple). Returns the original image with a color masked overlay
# AND a black and white mask with nothing behind it.
def process_image(image_path, annotations_df, color_map):
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create masked version
    masked_img = img.copy()
    
    # Apply all masks for this image
    image_annotations = annotations_df[annotations_df['filename'] == os.path.basename(image_path)]
    for _, row in image_annotations.iterrows():
        points_x = row['points']['all_points_x']
        points_y = row['points']['all_points_y']
        defect_class = row['attributes']['Defect_Class']
        
        if defect_class in color_map:
            color = color_map[defect_class]
            color_bgr = [int(x * 255) for x in color[:3]]
            masked_img = apply_mask(masked_img, np.column_stack((points_x, points_y)), color_bgr)

    return img_rgb, masked_img


def main(input_folder, csv_path, mask_output_folder, overlay_output_folder):
    # Create output folders if they don't exist
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(overlay_output_folder, exist_ok=True)
    
    # Load annotations and create color map
    annotations_df = load_annotations(csv_path)
    color_map = create_color_map()
    
    # Process all images
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_folder, filename)
            
            # Process image
            orig_img, masked_img = process_image(image_path, annotations_df, color_map)
            
            # Save outputs
            basename = os.path.splitext(filename)[0]
            
            # Save black and white mask only
            bw_mask = np.zeros_like(orig_img, dtype=np.uint8)
            image_annotations = annotations_df[annotations_df['filename'] == os.path.basename(image_path)]
            for _, row in image_annotations.iterrows():
                points_x = row['points']['all_points_x']
                points_y = row['points']['all_points_y']
                cv2.fillPoly(bw_mask, [np.column_stack((points_x, points_y))], (255, 255, 255))
            
            cv2.imwrite(os.path.join(mask_output_folder, f"{basename}_mask.png"), 
                       cv2.cvtColor(bw_mask, cv2.COLOR_RGB2BGR))
            
            # Save overlay version
            cv2.imwrite(os.path.join(overlay_output_folder, f"{basename}_overlay.png"), 
                       cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

# Folder paths. Change these to your specific folder.
INPUT_FOLDER = ''
CSV_PATH = ''
MASK_OUTPUT_FOLDER = ''
OVERLAY_OUTPUT_FOLDER = ''

# Run the script
if __name__ == "__main__":
    main(INPUT_FOLDER, CSV_PATH, MASK_OUTPUT_FOLDER, OVERLAY_OUTPUT_FOLDER)