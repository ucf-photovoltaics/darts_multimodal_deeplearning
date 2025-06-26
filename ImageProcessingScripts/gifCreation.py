from PIL import Image
import os
import re

# Extract the timestamp and module id because that is what's being used to identify the whole module. Takes in a filename. Returns a dictionary.
# This function will need to be turned into a more modular system eventually.
def extract_info(filename):
    """Extract timestamp and module id from filename."""
    match = re.search(r'(\d{8})_(\d{6})_(\w+)_stitched\.png', filename)
    if match:
        return {
            'timestamp': f"{match.group(1)}_{match.group(2)}",
            'module_id': match.group(3)
        }
    return None

# Creates the gifs from a folder of input images. Outputs the gif to the output folder. The input duration controls how many miliseconds are in between each image.
def create_gif_from_stitched_images(input_folder, output_folder, duration=500):
    # Get list of all stitched images
    image_files = [f for f in os.listdir(input_folder) 
                  if f.endswith('_stitched.png')]
    
    if not image_files:
        print("No stitched images found in the folder.")
        return
    
    # Group images by module id
    modules = {}
    for image_file in image_files:
        info = extract_info(image_file)
        if info:
            module_id = info['module_id']
            if module_id not in modules:
                modules[module_id] = []
            modules[module_id].append((image_file, info['timestamp']))
    
    # Create GIF for each module
    for module_id, image_list in modules.items():
        print(f"\nProcessing module: {module_id}")
        
        # Sort images by timestamp
        image_list.sort(key=lambda x: x[1])
        
        # Load images
        images = []
        for image_file, _ in image_list:
            img_path = os.path.join(input_folder, image_file)
            try:
                img = Image.open(img_path)
                # Convert to RGB format (required for GIF)
                img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Error loading {image_file}: {str(e)}")
                continue
        
        # Create GIF if we have images
        if images:
            # Create output filename with module id
            output_path = os.path.join(output_folder, f"{module_id}.gif")
            
            # Save as GIF
            images[0].save(
                output_path,
                format='GIF',  # Explicitly specify GIF format
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0  # Infinite loop
            )
            print(f"GIF created successfully: {output_path}")
        else:
            print(f"No valid images loaded for module {module_id}")

if __name__ == "__main__":
    # Absolute filepaths for the input (individual cells) and the output (the final gif).
    input_folder = ""
    output_folder = ""
    create_gif_from_stitched_images(input_folder, output_folder)