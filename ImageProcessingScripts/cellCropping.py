from PIL import Image
import os

# Creates new filename with cell number cell_## at the end where ## represents 0 through (n-1) where n = columns * rows or the total number of cells.
def create_cell_filename(original_filename, cell_number):
    base, extension = os.path.splitext(original_filename)
    return f"{base}_cell{str(cell_number)}{extension}"

# Converts cell number to row and column coordinates.
def get_cell_position(cell_number, columns):
    return cell_number % columns, cell_number // columns

# Crops each image into the appropriate number of cells based on the rows and columns. Takes in the input folderpath (str), the output folderpath (str), the number of rows (int), and the number of columns (int).
def divide_images(input_folder, output_folder, rows, columns):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files in input folder
    image_files = [f for f in os.listdir(input_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Process each image file
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        try:
            # Open the image
            with Image.open(img_path) as img:
                width, height = img.size

                # Calculate cell dimensions
                cell_width = width // columns
                cell_height = height // rows

                # Create cells
                for cell_num in range(rows * columns):
                    col, row = get_cell_position(cell_num, columns)

                    # Calculate crop box coordinates
                    left = col * cell_width
                    top = row * cell_height
                    right = left + cell_width
                    bottom = top + cell_height

                    # Crop and save the cell
                    cell_img = img.crop((left, top, right, bottom))
                    new_filename = create_cell_filename(filename, cell_num)
                    cell_img.save(os.path.join(output_folder, new_filename))

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

def main():
    INPUT_FOLDER = ""
    OUTPUT_FOLDER = ""
    ROWS = 3
    COLUMNS = 12

    divide_images(INPUT_FOLDER, OUTPUT_FOLDER, ROWS, COLUMNS)

if __name__ == "__main__":
    main()
