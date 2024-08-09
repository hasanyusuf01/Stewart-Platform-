
import cv2
import numpy as np
import os
import csv

'''
t---> for target image
c---> for capturing image at a state 
d---> for capturing duplicate image at of a state 


'''

# Calibration data
calib_data_path = "C:\\Users\\Azad Singh\\Desktop\\New folder\\calib_data\\MultiMatrix.npz"
calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coeff = calib_data["distCoef"]
r_vector = calib_data["rVector"]
t_vector = calib_data["tVector"]
cap = cv2.VideoCapture(0)

# Output directory
output_dir = "C:\\Users\\Azad Singh\\Desktop\\output\\D10"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# CSV file path
csv_file_path = os.path.join(output_dir, "image_data.csv")

# Create CSV file and write header if it doesn't exist
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Image_Name", "x", "y", "z", "roll", "pitch", "yaw"])

# Counter for image filenames
img_counter = 0
c=img_counter

while True:
    ret, frame = cap.read()
    # s_frame=frame
    if not ret:
        print("Camera not detected.")
        break
    frame_copy = frame.copy()

    # Drawing lines on the frame
    cv2.line(frame, (0, (frame.shape[0] // 2) - 209), (frame.shape[1], (frame.shape[0] // 2) - 209), (0, 0, 255), 2)  # x-axis (blue)
    cv2.line(frame, ((frame.shape[1] // 2) - 119, 0), ((frame.shape[1] // 2) - 119, frame.shape[0]), (0, 255, 0), 2)  # y-axis (green)

    cv2.imshow("Merged Frames", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('d'):
        z = roll = pitch = yaw = 0  # Set other DOF values to 0
        name = f"img{img_counter-1:03d}{c:01d}.jpg"
        c+=1

        # Save the frame without axis lines
        img_name = os.path.join(output_dir, name)
        cv2.imwrite(img_name, frame_copy)
        print(f"{img_name} saved!")
        # Save the image name and DOF values to the CSV file
        with open(csv_file_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # csv_writer.writerow([name, x, y, z, roll, pitch, yaw])
            csv_writer.writerow([name])
    if key == ord('t'):

        name = "target.jpg"

        # Save the frame without axis lines
        img_name = os.path.join(output_dir, name)
        cv2.imwrite(img_name, frame_copy)
        print(f"{img_name} saved!")
      
    elif key == ord('c'):
        # Prompt for x and y values
        # x = float(input("Enter value for x: "))
        # y = float(input("Enter value for y: "))
        z = roll = pitch = yaw = 0  # Set other DOF values to 0
        name = f"img{img_counter:03d}.jpg"

        # Save the frame without axis lines
        img_name = os.path.join(output_dir, name)
        cv2.imwrite(img_name, frame_copy)
        print(f"{img_name} saved!")
        img_counter += 1



        # Save the image name and DOF values to the CSV file
        with open(csv_file_path, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([name, 0, 0, z, roll, pitch, yaw])

cap.release()
cv2.destroyAllWindows()
