import cv2
import os
import numpy as np

# Adjusts the gamma of the image. Takes in the image, and the gamma value (int).
def adjust_gamma(image, gamma=3.0):
    """
    Apply gamma correction to an image.
    """
    # Create lookup table
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction
    return cv2.LUT(image, table)


# Proccesses all iamges in a folder. Takes in the input folderpath (str), and the output folderpath(str).
def process_images(input_folder, output_folder):
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Process each image
    for filename in os.listdir(input_folder):
        # Check if file is an image
        if os.path.splitext(filename)[1].lower() in image_extensions:
            img_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Read and process image
                image = cv2.imread(img_path)
                adjusted_image = adjust_gamma(image)

                # Save result
                cv2.imwrite(output_path, adjusted_image)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def main():
    input_folder = ""
    output_folder = ""

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process images
    process_images(input_folder, output_folder)

if __name__ == "__main__":
    main()
