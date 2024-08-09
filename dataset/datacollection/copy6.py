from PIL import Image
import os

def duplicate_image(file_path, output_dir, start_index, num_copies):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_image = Image.open(file_path)
    base_name = "img_c"
    for i in range(start_index, start_index + num_copies):
        new_filename = f"{base_name}{i:04d}.jpg"
        new_filepath = os.path.join(output_dir, new_filename)
        original_image.save(new_filepath)
        print(f"Saved {new_filepath}")

# Parameters
file_path = 'C:\\Users\\Azad Singh\\Desktop\\output\\D3\\img002.jpg'  # Path to the original image
output_dir = 'C:\\Users\\Azad Singh\\Desktop\\output\\D3'  # Directory to save duplicated images
start_index = 20  # Starting index for naming the duplicated images
num_copies = 3  # Number of copies to create

# Call the function
duplicate_image(file_path, output_dir, start_index, num_copies)
