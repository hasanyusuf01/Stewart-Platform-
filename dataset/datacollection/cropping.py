import cv2
import numpy as np
import os

def process_image(image_path, output_folder):
    # Load the image
    image = cv2.imread(image_path)
    h=285
    w=285
    y=3
    x=127
    # Crop the image to the bounding box of the largest contour
    cropped_image = image[y:y+h, x:x+w]

    
    # Save the cropped grid image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"{filename}")
    cv2.imwrite(output_path, cropped_image)
    print(f"Processed {filename}")

inp="D9"
# Folder containing the images
input_folder = "C:\\Users\\Azad Singh\\Desktop\\output\\raw\\"+inp

output_folder = "C:\\Users\\Azad Singh\\Desktop\\output\\processed\\"+inp

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        process_image(image_path, output_folder)
