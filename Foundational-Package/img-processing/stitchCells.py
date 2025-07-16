import os
from PIL import Image
import re

__all__ = ["get_cell_number", "extract_module_name", "create_module", "stitch_all_modules"]
# Right now this script only works for 6x10 modules. This function should be adapted to work for any sized module so long as the correct cell x cell dimensions are given.

# Extracts the cell number from the filename.
def get_cell_number(filename):
    # Try to match both formats: cell_## and cell_#
    match = re.search(r'cell_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None

# Extract the module name from the filename and return the module name.
def extract_module_name(filename):
    base_name = filename.rsplit('_', 2)[0]
    return base_name

# Stitches the cells together in a single modules. Takes in the folder path to the cells, the extracted filename, a list of the images involved, and the output directory.
def create_module(images_dir, module_filename, images_list, output_dir, rows, columns):
    # print(f"\n=== Processing module: {module_filename} ===")
    # print(f"DEBUG: Current working directory: {os.getcwd()}")
    # print(f"DEBUG: Images directory: {images_dir}")
    # print(f"DEBUG: Output directory: {output_dir}")
    
    # Verify we have the expected amount of images
    expected_count = rows * columns
    if len(images_list) != expected_count:
        print(f"WARNING: Expected {expected_count} cells but found {len(images_list)}")
        print("Files:", images_list)
        return None
        
    # Validate all filenames belong to the same module
    base_name = extract_module_name(images_list[0])
    invalid_files = [f for f in images_list if extract_module_name(f) != base_name]
    
    if invalid_files:
        print("\nERROR: Files from different modules detected!")
        print(f"Base module name: {base_name}")
        print("Invalid files:")
        for f in invalid_files:
            print(f"- {f}")
        return None
    
    # Verify cell numbers are sequential
    cell_numbers = [get_cell_number(f) for f in images_list]
    expected_numbers = list(range(expected_count))
    
    if sorted(cell_numbers) != expected_numbers:
        print("\nERROR: Cell numbers are not sequential!")
        print(f"Expected numbers: {expected_numbers}")
        print(f"Found numbers: {sorted(cell_numbers)}")
        return None
        
    # print("\nValidation successful! Creating module...")
    
    # print("\nLoading images...")
    # Load all images
    images = []
    for filename in images_list:
        img_path = os.path.join(images_dir, filename)
        # print(f"DEBUG: Attempting to load: {img_path}")
        try:
            img = Image.open(img_path)
            images.append(img)
        #    print(f"DEBUG: Successfully loaded {filename} - Size: {img.size}")
        except Exception as e:
            print(f"ERROR: Could not load {filename}: {str(e)}")
            return None
            
    # Get dimensions
    width = images[0].width
    height = images[0].height
    # print(f"\nCreating canvas: {width * columns}x{height * rows}")
    
    # Create large canvas
    canvas = Image.new('RGB', (width * columns, height * rows))
    
    # Place images in grid according to specified pattern
    for i in range(expected_count):
        # Calculate position in the grid
        row = i // columns
        col = i % columns   
        
        # Place image on canvas
        x_pos = col * width
        y_pos = row * height
        canvas.paste(images[i], (x_pos, y_pos))
        
        # Print progress every row
        #if i % 10 == 9:  # After completing each row
        #    print(f"DEBUG: Placed row {row + 1}/6")
    
    # Save combined image
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, module_filename)
    # print(f"DEBUG: Saving to: {output_path}")
    canvas.save(output_path)
    print(f"Processed: {module_filename} and stitched them together")
    return output_path

# Runs the other scripts for all of the folders in a directory. This should probably be turned into its own function that is not dependent on the rest of these scripts but is modularized to accept other forms of image processing.
def stitch_all_modules(images_dir, output_dir, rows=6, columns=10):
    # Get all files matching pattern
    files = [f for f in os.listdir(images_dir) if re.search(r'^.*?cell_\d+\.png$', f)]
    # print(f"DEBUG: Found {len(files)} potential files:")
    # for f in files[:5]:  # Show first 5 files for verification
    #     print(f"- {f}")
    # if len(files) > 5:
    #     print("- ...")  # Indicate there might be more
    
    # Group files by module name
    modules = {}
    for filename in files:
        module_name = extract_module_name(filename)
        cell_num = get_cell_number(filename)
        
        if module_name not in modules:
            modules[module_name] = []
            
        modules[module_name].append((filename, cell_num))
    
    # Sort cells within each module
    for module_name in modules:
        modules[module_name].sort(key=lambda x: x[1])
        modules[module_name] = [x[0] for x in modules[module_name]]
    
    # Process each module
    for module_num, (module_name, files_list) in enumerate(modules.items(), 1):
        # Create module filename (e.g., "module_001.png")
        module_filename = f"{module_name}_stitched.png"
        
        # print(f"\n{'='*50}")
        # print(f"Processing module {module_num}/{len(modules)}: {module_name}")
        # print(f"Files: {files_list[:5]}...")
        # if len(files_list) > 5:
        #     print("...")
        
        try:
            output_path = create_module(images_dir, module_filename, files_list, output_dir, rows, columns)
            # print(f"{'='*50}\n")
        except Exception as e:
            print(f"ERROR processing module {module_name}: {str(e)}\n")
            
    print(f"Stitched {len(modules)} images in '{images_dir}' and saved to '{output_dir}'")
