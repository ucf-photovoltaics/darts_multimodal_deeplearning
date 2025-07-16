import os
import pandas as pd
import json

# Define paths to image and JSON folders
image_folder = '/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/UCF-EL-Defect/M55 EL Data/3x3/Cells'
json_folder = '/mnt/vstor/CSE_MSE_RXF131/cradle-members/sdle/jmr375/UCF-EL-Defect/M55 EL Data/3x3/Defects'

# Initialize lists to store file information
image_files = []
json_data = []

# Search through image folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        image_files.append(filename)

# Search through JSON folder
for filename in os.listdir(json_folder):
    if filename.endswith('.json'):
        json_file_path = os.path.join(json_folder, filename)
        try:
            with open(json_file_path, 'r') as f:
                json_data_dict = json.load(f)
                crack = json_data_dict['crack']
                contact = json_data_dict['contact']
                interconnect = json_data_dict['interconnect']
                corrosion = json_data_dict['corrosion']
                row = [filename, crack, contact, interconnect, corrosion, '']
                json_data.append(row)
        except json.JSONDecodeError:
            print(f"Error decoding JSON file: {filename}")


# Sort the json_data list based on filenames
image_files = sorted(image_files) #correct

for row in json_data:
    filename = row[0]
    cellNum = filename.split('cell')[1].split('.json')[0]
    cellNum = int(cellNum)
    row[5] = cellNum

json_data = sorted(json_data, key=lambda x: x[5]) #incorrect
print(json_data[:5])

dataList = []
dataList.append(['Filename', 'Cellname', 'Crack', 'Contact', 'Interconnect', 'Corrosion'])
for filename, row in zip(image_files, json_data):
    cellname = row[0]
    crack = row[1]
    contact = row[2]
    interconnect = row[3]
    corrosion = row[4]
    newRow = [filename, cellname, crack, contact, interconnect, corrosion]
    dataList.append(newRow)

# Create dataframe with image filenames as first column
df = pd.DataFrame(dataList)

print(df)

#df['Cellname'] = df['Cellname'].apply(lambda x: f"cell{x}" if isinstance(x, int) else x)
#print(df)

# Save the dataframe to a CSV file (optional)
df.to_csv('M55-3x3Dataframe.csv', index=False)

