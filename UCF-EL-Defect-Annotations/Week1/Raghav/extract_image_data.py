import os
import pandas as pd

def main():
    """
    Create a DataFrame with the exact structure shown in the example.
    """
    # Sample data matching the example
    data = [
        # First image with 8 regions
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_2.png", "file_size": 106269, "file_attributes": "{}", "region_count": 8, "region_id": 0, "region_shape_attributes": '{"name":"polygon","all_points_x":[42,42,50,50],"all_points_y":[5,15,15,5]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_2.png", "file_size": 106269, "file_attributes": "{}", "region_count": 8, "region_id": 1, "region_shape_attributes": '{"name":"polygon","all_points_x":[230,231,242,242],"all_points_y":[10,20,20,10]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_2.png", "file_size": 106269, "file_attributes": "{}", "region_count": 8, "region_id": 2, "region_shape_attributes": '{"name":"polygon","all_points_x":[295,226,177,61,50,120,182,295],"all_points_y":[30,40,50,60,70,80,90,30]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_2.png", "file_size": 106269, "file_attributes": "{}", "region_count": 8, "region_id": 3, "region_shape_attributes": '{"name":"polygon","all_points_x":[189,174,161,60,53,120,156,189],"all_points_y":[35,45,55,65,75,85,95,35]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_2.png", "file_size": 106269, "file_attributes": "{}", "region_count": 8, "region_id": 4, "region_shape_attributes": '{"name":"polygon","all_points_x":[65,65,56,56],"all_points_y":[25,35,35,25]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_2.png", "file_size": 106269, "file_attributes": "{}", "region_count": 8, "region_id": 5, "region_shape_attributes": '{"name":"polygon","all_points_x":[151,152,147,146],"all_points_y":[40,50,50,40]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_2.png", "file_size": 106269, "file_attributes": "{}", "region_count": 8, "region_id": 6, "region_shape_attributes": '{"name":"polygon","all_points_x":[109,109,116,115],"all_points_y":[45,55,55,45]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_2.png", "file_size": 106269, "file_attributes": "{}", "region_count": 8, "region_id": 7, "region_shape_attributes": '{"name":"polygon","all_points_x":[80,86,85,81],"all_points_y":[50,60,60,50]}'},
        
        # Second image with 8 regions
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_3.png", "file_size": 106657, "file_attributes": "{}", "region_count": 8, "region_id": 0, "region_shape_attributes": '{"name":"polygon","all_points_x":[48,53,60,59],"all_points_y":[6,16,16,6]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_3.png", "file_size": 106657, "file_attributes": "{}", "region_count": 8, "region_id": 1, "region_shape_attributes": '{"name":"polygon","all_points_x":[72,74,79,78],"all_points_y":[10,20,20,10]}'},
        {"filename": "20170412_152026_JABIL_QCELLS_521011165014200055_0Pa_s_A_V_cell_3.png", "file_size": 106657, "file_attributes": "{}", "region_count": 8, "region_id": 2, "region_shape_attributes": '{"name":"polygon","all_points_x":[133,133,147,146],"all_points_y":[30,40,40,30]}'}
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'image_attributes.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved image attributes to: {csv_path}")
    
    # Display the data in the console
    print("\nImage Attributes (first few rows):")
    print(df)

if __name__ == "__main__":
    main()
