from rembg import remove
from PIL import Image
import os
import numpy as np

__all__ = ["remove_background", "process_folder"]

# Removes the background of a given image and then crops it before saving. Takes in the input filepath (str) for the image the output filepath (str) for where to save it to and the threshold value (int: 0-255). The threshold is actually used here where it was just being passed to this function from the process_folder function.
def remove_background(input_path, output_path, threshold=10):
    try:
        # Open the input image
        with Image.open(input_path) as input_image:
            # Remove the background
            output = remove(input_image)

            # Convert to RGBA for manual cropping
            rgba = np.array(output)

            # Create a mask for non-transparent pixels
            # Using the alpha channel (last channel)
            alpha_channel = rgba[:,:,3]

            # Find pixels that are not completely transparent
            # Using the threshold value
            mask = alpha_channel > threshold

            # Get the bounding box coordinates
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Crop the image
            if rmin is not None and rmax is not None and cmin is not None and cmax is not None:
                cropped = output.crop((cmin, rmin, cmax + 1, rmax + 1))
                output = cropped

            # Save the result
            # Uses PNG format for transparency
            if not output_path.lower().endswith(('.png', '.tiff', '.bmp')):
                output_path = output_path.rsplit('.', 1)[0] + '.png'

            output.save(output_path)
            print(f"Processed: {os.path.basename(input_path)} and removed the background")

    except Exception as e:
        print(f"Error processing image: {str(e)}")

# Processes each image in the folder. Takes in the input folder filepath (str), the output folder filepath (str) and a threshold (int: 0-255) for removing the background (too high removes the module too!).
def process_folder(input_folder, output_folder, threshold=10):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            # Convert output filename to PNG if needed
            output_filename = filename.rsplit('.', 1)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            remove_background(input_path, output_path, threshold)
            
    print(f"Removed the background of {len(os.listdir(input_folder))} images in '{input_folder}' and saved to '{output_folder}'")
