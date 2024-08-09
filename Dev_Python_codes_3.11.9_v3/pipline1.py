
import sys

# Add the current directory to sys.path
sys.path.append('.')
import numpy as np
import cv2 
# import keras 
# from keras.models import load_model
from robot import *
import numpy as np
import math

import torch
# import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import os
# import numpy as np
# from PIL import Image
# import torch
from torchvision import models




''' 
# 
# Overview
# The designed pipeline for robot movement incorporates live feed acquisition, model prediction,
# conversion of model output to 6 Degrees of Freedom (6DOF), and transmission of this data to the robot.
# This report details the flow of function calls and their respective roles in the pipeline.
# 
# Pipeline Flow
# 
# 1. Live Feed Acquisition:
#     - Function: capture_live_video() or read_video_file(path_to_video)
#     - Purpose: To obtain the video feed, either through live streaming or by reading a video file.
# 
# 2. Frame Processing and Model Prediction:
#     - Function: model_predict(frame)
#     - Purpose: To process each frame from the video feed and predict the 6DOF parameters.
#     - Output: E = (x, q)
#         - x = [Δx, Δy, Δz]
#         - q = [q0, q1, q2, q3]
# 
# 3. Conversion to Euler Angles:
#     - Function: euler_from_quaternion(quaternion)
#     - Purpose: To convert quaternion output q to Euler angles for roll, pitch, and yaw.
#     - Output: Euler angles corresponding to the quaternion.
# 
# 4. Robot Control:
#     - Function: robot_controll(predictions)
#     - Purpose: To convert the 6DOF parameters to the format [Δx, Δy, Δz, roll, pitch, yaw] 
#                and send these commands to the robot.
# 
# Detailed Function Flow
# 
# 1. Capture Live Video or Read Video File:
#     - The initial step involves either capturing live video using capture_live_video() 
#       or reading from a pre-recorded file using read_video_file(path_to_video). 
#       This function continuously feeds frames to the next stage.
# 
# 2. Model Prediction:
#     - Each frame obtained from the previous step is input to the model_predict(frame) function.
#       This function utilizes a deep learning model to predict the 6DOF parameters, 
#       providing an estimate of the robot's pose and orientation in the form E = (x, q).
# 
# 3. Quaternion to Euler Conversion:
#     - The quaternion part of the model's output, q, is passed to euler_from_quaternion(quaternion). 
#       This conversion is crucial for interpreting the robot's orientation in terms of roll, pitch, 
#       and yaw angles, which are more intuitive for control purposes.
# 
# 4. Sending Commands to the Robot:
#     - The final stage involves the robot_controll(predictions) function. 
#       This function converts the 6DOF parameters [Δx, Δy, Δz] and the Euler angles (roll, pitch, yaw) 
#       into a command format that the robot can interpret and execute. 
#       These commands are then transmitted to the robot, enabling it to adjust its position and 
#       orientation accordingly.


'''
output_dir = "C:\\Users\\Azad Singh\\Desktop\\yusuf\\output\\D10"
targetsize = (200, 200)
kernel = np.ones((3,3), np.uint8)
device = torch.device('cpu')
img_name = os.path.join(output_dir, "pp.jpg")
# Existing functions (copy all the functions here)
# preprocess_input_img, load_image, model_predict, robot_controll, CustomResNet

def preprocess_input_img(test_image_path):
    orig_sample_test_img = cv2.cvtColor(test_image_path, cv2.COLOR_BGR2RGB)
    x = 177
    y = 3
    width = 200
    height = 200
    orig_sample_test_img = orig_sample_test_img[y:y+height, x:x+width]
    gray_sample_test_img = cv2.cvtColor(orig_sample_test_img, cv2.COLOR_RGB2GRAY)
    gray_resized_test_img = cv2.resize(gray_sample_test_img, targetsize,
                        interpolation = cv2.INTER_AREA)   # To shrink an image
    (thresh, black_n_white_sample_img) = cv2.threshold(gray_resized_test_img, 80,255, cv2.THRESH_BINARY_INV)
#     black_n_white_sample_img =cv2.GaussianBlur(black_n_white_sample_img , (3, 3), 0)
    black_n_white_sample_img= cv2.dilate(black_n_white_sample_img, kernel, iterations=1)
    _, black_n_white_sample_img = cv2.threshold(black_n_white_sample_img, 50, 255, cv2.THRESH_BINARY)
    cv2.imwrite(img_name, black_n_white_sample_img)
    black_n_white_sample_img = black_n_white_sample_img/255
    return black_n_white_sample_img

def load_image(image, target_size=targetsize):
    img = preprocess_input_img(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img)
    img = img.float()

    return img



'''----------------------------------------------------------------------------------------------------'''
def model_predict(frame,target):
    resized_frame = cv2.resize(frame, (200, 200))
    target_image = load_image(target).unsqueeze(0).to(device)
    test_image = load_image(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = outputs = model(target_image, test_image)
    Δx = []
    Δy = []
    Δx.extend(predictions[:, 0].cpu().numpy())
    Δy.extend(predictions[:, 1].cpu().numpy())
    roll = pitch = yaw = Δz = 0
    if Δx[0] > 0.5:
        Δx[0] = 1
    if Δx[0] < -0.5:
        Δx[0] = -1
    if -0.5 < Δx[0] < 0.5:
        Δx[0] = 0
    if Δy[0] > 0.5:
        Δy[0] = 1
    if Δy[0] < -0.5:
        Δy[0] = -1
    if -0.5 < Δy[0] < 0.5:
        Δy[0] = 0
    return np.array(Δx), np.array(Δy), Δz, roll, pitch, yaw
'''----------------------------------------------------------------------------------------------------'''

def robot_controll(Δx, Δy ,Δz,roll,pitch,yaw):

    print("robot is moving where Δx, Δy ,Δz,roll,pitch,yaw ==",Δx[0], Δy[0] ,Δz,roll,pitch,yaw)
    x_=int(Δx[0]*30)
    y_=int(Δy[0]*30)
    rbt.move_abs(x=x_,y=y_ )


'''----------------------------------------------------------------------------------------------------'''
def capture_live_video():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    target_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not detected.")
            break

        frame_copy = frame.copy()
        cv2.line(frame, (0, (frame.shape[0] // 2) - 209), (frame.shape[1], (frame.shape[0] // 2) - 209), (0, 0, 255), 2)
        cv2.line(frame, ((frame.shape[1] // 2) - 119, 0), ((frame.shape[1] // 2) - 119, frame.shape[0]), (0, 255, 0), 2)

        cv2.imshow("Merged Frames", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if key == ord('t'):
            target_image = frame_copy
            print("Target image captured!")

        elif key == ord('c'):
            if target_image is not None:
                captured_image = frame_copy
                Δx, Δy, Δz, roll, pitch, yaw = model_predict(captured_image, target_image)
                robot_controll(Δx, Δy, Δz, roll, pitch, yaw)
            else:
                print("No target image available. Press 't' to capture a target image first.")

    cap.release()
    cv2.destroyAllWindows()
'''----------------------------------------------------------------------------------------------------'''
def read_video_file(video_file):
    # Read a video file
    cap_file = cv2.VideoCapture(video_file)

    # Check if the video file is opened successfully
    if not cap_file.isOpened():
        print("Error: Could not open video file")
        return

    while True:
        # Read a frame from the video file
        ret_file, frame_file = cap_file.read()
        # predict 6DOF on the  video feed
        Δx, Δy ,Δz,roll,pitch,yaw= model_predict(frame_file)
        robot_controll(Δx, Δy ,Δz,roll,pitch,yaw)
        # Display the video file
        # cv2.imshow('Video File', frame_file)

        # Wait for user input to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video file
    cap_file.release()
    cv2.destroyAllWindows()
'''----------------------------------------------------------------------------------------------------'''


        

resnet = models.resnet18(pretrained=True)

count = 0
for child in resnet.children():
  count+=1
  if count < 7:
    for param in child.parameters():
        param.requires_grad = False



num_ftrs = resnet.fc.in_features
#num_ftrs

resnet.fc = nn.Identity()

class CustomResNet(nn.Module):
    def __init__(self, num_outputs=2):
        super(CustomResNet, self).__init__()

        
        self.resnet = resnet
        self.c1 = nn.Conv2d(1, 3, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=False)

        self.fc1 = nn.Linear(num_ftrs * 2, 512)
        self.r = nn.ReLU(inplace=True)
        self.fc11 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, num_outputs   )
        self.fc8 = nn.Linear(8, 1   )
        self.d = nn.Dropout(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x1, x2):
        x1 = self.c1(x1)
        x2 = self.c1(x2)

        f1 = self.resnet(x1)
        f2 = self.resnet(x2)
        combined = torch.cat((f1, f2), dim=1)
        x = nn.ReLU()(self.fc1(combined))
        x = self.d(x)
        x = nn.ReLU()(self.fc11(x))
        x = self.d(x)
        x = nn.ReLU()(self.fc2(x))
        x = self.d(x)
        x = nn.ReLU()(self.fc3(x))
        x = self.d(x)
        x = nn.ReLU()(self.fc4(x))
        x = self.d(x)
        x = nn.ReLU()(self.fc5(x))
        x = self.d(x)
        x = nn.ReLU()(self.fc6(x))
        y = self.tanh(self.fc7(x))
#         z = self.tanh(self.fc8(x))
        return y


# Initialize the model
# model = CustomResNet()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# Load the trained modelC:\\Users\\Azad Singh\\Desktop\\output\\10resnet_.pth  "C:\Users\Azad Singh\Desktop\models.pth\results_2024_07_11-15_32_56best_model.pt"
model_path ="C:\\Users\\Azad Singh\\Desktop\\models_pth\\results18_2024_07_11-15_32_56resnet_last.pth"
model = CustomResNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)



rbt = Robot()
rbt.connect('com3')
capture_live_video()
