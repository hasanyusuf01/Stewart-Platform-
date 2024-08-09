import cv2
import streamlit as st
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import os
import time
from robot import *

# Your existing code (copy here)
output_dir = "C:\\Users\\Azad Singh\\Desktop\\output\\D10"
targetsize = (200, 200)
kernel = np.ones((3,3), np.uint8)
device = torch.device('cpu')

# Existing functions (copy all the functions here)
# preprocess_input_img, load_image, model_predict, robot_controll, CustomResNet

def preprocess_input_img(test_image_path):
    orig_sample_test_img = cv2.cvtColor(test_image_path, cv2.COLOR_BGR2RGB)
    x = 50
    y = 0
    width = 200
    height = 200
    orig_sample_test_img = orig_sample_test_img[y:y+height, x:x+width]
    gray_sample_test_img = cv2.cvtColor(orig_sample_test_img, cv2.COLOR_RGB2GRAY)
    gray_resized_test_img = cv2.resize(gray_sample_test_img, targetsize,
                        interpolation = cv2.INTER_AREA)   # To shrink an image
    (thresh, black_n_white_sample_img) = cv2.threshold(gray_resized_test_img, 70,255, cv2.THRESH_BINARY_INV)
#     black_n_white_sample_img =cv2.GaussianBlur(black_n_white_sample_img , (3, 3), 0)
    black_n_white_sample_img= cv2.dilate(black_n_white_sample_img, kernel, iterations=1)
    _, black_n_white_sample_img = cv2.threshold(black_n_white_sample_img, 50, 255, cv2.THRESH_BINARY)
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


class CustomResNet(nn.Module):
    def __init__(self, num_outputs=6):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.c1 =nn.Conv2d(1, 8, kernel_size=(3, 3),stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet.conv1 =nn.Conv2d(8, 64, kernel_size=(3, 3),stride=(2, 2), padding=(3, 3), bias=False)
        
        
        
        num_ftrs = self.resnet.fc.in_features
#         print(num_ftrs)
#         self.resnet.avgpool=nn.Identity()
        self.resnet.fc = nn.Identity()  # Remove the last fully connected layer

        # # Additional layers for combined features
#         self.fc0=nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.fc1 = nn.Linear(num_ftrs * 2, 512)
        self.r=nn.ReLU(inplace=True)
        self.fc11= nn.Linear(512,256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 2)
#         self.fc8= nn.Linear(4,2)
        self.d = nn.Dropout(0.2)
        self.tanh = nn.Tanh()

    def forward(self, x1, x2):
        # Extract features from both images
        x1=self.c1(x1)
        x2=self.c1(x2)

        f1 = self.resnet(x1)
        f2 = self.resnet(x2)
        # Concatenate features
        combined = torch.cat((f1, f2), dim=1)
        # Fully connected layers
        x = nn.ReLU()(self.fc1(combined))
        x= self.d(x)
        x = nn.ReLU()(self.fc11(x))
        x= self.d(x)
        x = nn.ReLU()(self.fc2(x))
        x= self.d(x)
        x = nn.ReLU()(self.fc3(x))
        x= self.d(x)
        x = nn.ReLU()(self.fc4(x))
        x= self.d(x)
        x = nn.ReLU()(self.fc5(x))
        x= self.d(x)
        x = nn.ReLU()(self.fc6(x))
        z = self.tanh(self.fc7(x))
        return z

def model_predict(frame, target):
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

def robot_controll(Δx, Δy, Δz, roll, pitch, yaw):
       x_=int(Δx[0]*10)
       y_=int(Δy[0]*10)
       rbt.move_abs(x=x_,y=y_ )

    # st.write(f"Robot moving to Δx={Δx[0]}, Δy={Δy[0]}, Δz={Δz}, roll={roll}, pitch={pitch}, yaw={yaw}")
# "C:\Users\Azad Singh\Desktop\output\results_2024_07_10-07_09_18best_model.pt"
model_path = 'C:\\Users\\Azad Singh\\Desktop\\output\\results_2024_07_10-07_09_18best_model.pt'
model = CustomResNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)


# Streamlit code
st.title("Robot Movement Prediction")
# st.title("Webcam Live Feed")
# _image = st.checkbox('Capture Frame and Predict')
# stop = st.checkbox('Stop Live Video')
output_container = st.empty()

run_mode = st.sidebar.selectbox("Select Run Mode", ["Live Video", "Video File"])

if run_mode == "Live Video":
    rbt = Robot()
    rbt.connect('com3')
    target_image = st.checkbox('Upload Target Image')
    if target_image==True:
        st.subheader("Upload Image for Prediction")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, -1)
            target_image = image
            st.subheader("Live Camera Feed")

            FRAME_WINDOW = st.image([])
            camera = cv2.VideoCapture(0)

            while run_mode:
                _, frame = camera.read()
                frame_copy = frame.copy()
                cv2.line(frame, (0, (frame.shape[0] // 2) - 209), (frame.shape[1], (frame.shape[0] // 2) - 209), (0, 0, 255), 2)
                cv2.line(frame, ((frame.shape[1] // 2) - 119, 0), ((frame.shape[1] // 2) - 119, frame.shape[0]), (0, 255, 0), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                captured_image = frame_copy
                Δx, Δy, Δz, roll, pitch, yaw = model_predict(captured_image, target_image)
                robot_controll(Δx, Δy, Δz, roll, pitch, yaw)

                out= "x:"+str( Δx[0])+" y:"+str( Δy[0])
                output_container.text(out)
                time.sleep(0.5)


else:
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="video_file")
    if video_file is not None:
        cap_file = cv2.VideoCapture(video_file.name)
        while True:
            ret_file, frame_file = cap_file.read()
            if not ret_file:
                break
            Δx, Δy, Δz, roll, pitch, yaw = model_predict(frame_file, frame_file)
            robot_controll(Δx, Δy, Δz, roll, pitch, yaw)
            st.image(frame_file, channels="BGR", caption="Video File Frame")
            if st.button("Stop Video", key="stop_video"):
                break