import pandas as pd
import os

def main():
    """
    Read the CSV file and display the data in a readable format
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the CSV file
    csv_path = os.path.join(current_dir, 'image_attributes_final.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Display summary information
    print(f"Total number of records: {len(df)}")
    print(f"Number of unique images: {df['filename'].nunique()}")
    print(f"Number of regions per image: {df.groupby('filename')['region_id'].count().iloc[0]}")
    
    # Display the first few records for each of the first 3 images
    unique_images = df['filename'].unique()[:3]
    
    for image in unique_images:
        print(f"\n\nImage: {image}")
        image_df = df[df['filename'] == image].head(3)
        
        for _, row in image_df.iterrows():
            print(f"\nRegion ID: {row['region_id']}")
            print(f"File Size: {row['file_size']} bytes")
            print(f"Region Count: {row['region_count']}")
            print(f"Region Shape Attributes: {row['region_shape_attributes']}")

if __name__ == "__main__":
    main()
