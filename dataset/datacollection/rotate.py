import os
import cv2
import pandas as pd

''' 
code for ardumenting data 
  --argumenting technique used-->rotaion clockwise by 90 degree 
  ------the csv file is mapped by the maping function
  ------the images are rotated by the rotate function
inp---> its the folder name on which the operation is to be performed 


'''

# Define mapping function
def map_values(row):
    x = row['x']
    y = row['y']
    
    # Define your mapping rules based on x and y values
    if x == 1 and y == 0:
        return pd.Series({'x': 0, 'y': -1})  # Replace x=1, y=0 with x=0, y=-1
    if x == 0 and y == -1:
        return pd.Series({'x': -1, 'y': 0})
    if x == 0 and y == 1:
        return pd.Series({'x': 1, 'y': 0})
    if x == -1 and y == 0:
        return pd.Series({'x': 0, 'y': 1})
    if x == -1 and y == -1:
        return pd.Series({'x': -1, 'y': 1})
    if x == 1 and y == -1:
        return pd.Series({'x': -1, 'y': -1})
    if x == -1 and y == 1:
        return pd.Series({'x': 1, 'y': 1})
    if x == 1 and y == 1:
        return pd.Series({'x': 1, 'y': -1})

    elif x == 0 and y == 0:
        return pd.Series({'x': 0, 'y': 0})  # Replace x=0, y=1 with x=-1, y=0
    # Add more conditions as needed
    
    # If no conditions match, return the original values
    return row



def rotate_image(image_path, output_folder):
    # Load the image
    image = cv2.imread(image_path)
    
    # Rotate the image by 90 degrees clockwise
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    # Get the filename from the image path
    filename = os.path.basename(image_path)
    
    # Create the full output path
    output_path = os.path.join(output_folder, filename)
    
    # Save the rotated image
    cv2.imwrite(output_path, rotated_image)
    print(f"Rotated and saved {filename}")

# Folder containing the images
inp="D9_1_2"
rootdir="C:\\Users\\Azad Singh\\Desktop\\output\\processed"


input_folder = os.path.join(rootdir, inp)

# Folder where the rotated images will be saved
output_folder = os.path.join(rootdir, f"{inp}_93")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(input_folder, filename)
        rotate_image(image_path, output_folder)

file_path = os.path.join(input_folder, "image_data.csv")
# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)
# Apply the mapping function to each row
df[['x', 'y']] = df.apply(map_values, axis=1)[['x', 'y']]

# Save the modified DataFrame back to CSV if needed
# Construct the path for the modified CSV file
modified_file_path = os.path.join(output_folder, 'image_data.csv')
# Save the modified DataFrame to CSV
df.to_csv(modified_file_path, index=False)