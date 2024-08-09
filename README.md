
---

# Stewart-Platform

This repository contains code for training custom multimodal neural networks designed to predict the 6 Degrees of Freedom (6DoF) for a parallel robot.

## Table of Contents

1. [Dataset Collection and Preprocessing Scripts](#dataset-collection-and-preprocessing-scripts)
   - [rotate.py](#rotatepy)
   - [data_collection.py](#data_collectionpy)
   - [cropping.py](#croppingpy)
   - [copy6.py](#copy6py)
   - [Requirements](#requirements)
   - [Usage](#usage)

2. [Custom Deep Learning Models for 6DoF Prediction](#custom-deep-learning-models-for-6dof-prediction)
   - [vgg.ipynb](#vggipynb)
   - [ResNet50_Yusuf.ipynb](#resnet50_yusufipynb)
   - [ResNet50-Copy1.ipynb](#resnet50-copy1ipynb)
   - [ResNet50.ipynb](#resnet50ipynb)
   - [ResNet18-pretrained.ipynb](#resnet18-pretrainedipynb)
   - [resnet.ipynb](#resnetipynb)
   - [posenet-Copy1.ipynb](#posenet-copy1ipynb)
   - [posenet.ipynb](#posenetipynb)
   - [custom.ipynb](#customipynb)
   - [Requirements](#requirements-1)
   - [Usage](#usage-1)

3. [Scripts for Robotic Vision and Tracking](#scripts-for-robotic-vision-and-tracking)
   - [track_arucoMarker.py](#track_arucomarkerpy)
   - [shape_detection.py](#shape_detectionpy)
   - [mimic_aruco.py](#mimic_arucopy)
   - [m.py](#mpy)
   - [countour_arucomarker.py](#countour_arucomarkerpy)
   - [colour_detection.py](#colour_detectionpy)
   - [aruco.py](#arucopy)
   - [Requirements](#requirements-2)
   - [Usage](#usage-2)

4. [Testing and Model Evaluation Scripts--Dev_Python_codes_3.11.9_v3](#testing-and-model-evaluation-scripts) 
   - [streamlit_app.py](#streamlit_apppy)
   - [pipline1.py](#pipline1py)
   - [Python Version Requirement](#python-version-requirement)

---

## Dataset Collection and Preprocessing Scripts

This folder contains scripts for collecting and preprocessing data for training deep learning models. These scripts handle tasks such as data augmentation, image cropping, and duplication to prepare datasets for effective model training.

### rotate.py

This script performs data augmentation by rotating images 90 degrees clockwise. It also updates the corresponding CSV files to reflect changes in the coordinate system after rotation.

### Key Functions

- **map_values(row):** Maps x and y coordinates to their new values after rotation.
- **rotate_image(image_path, output_folder):** Rotates an image and saves it to the specified output folder.

### data_collection.py

This script captures images from a video stream and saves them along with corresponding 6DoF values into a CSV file. It supports capturing target images and duplicate images at different states.

### Key Features

- **Target Image Capture:** Capture a reference image for the dataset.
- **Image Duplication:** Capture multiple images for a given state to increase dataset size.
- **CSV Logging:** Records image names and 6DoF values in a CSV file.

### cropping.py

This script processes images by cropping them to a specific size. It crops the images based on predefined coordinates and saves the processed images to an output directory.

### Key Function

- **process_image(image_path, output_folder):** Crops the image to specified dimensions and saves it.

### copy6.py

This script duplicates an image multiple times and saves the duplicates with incremented filenames. It is useful for augmenting datasets by increasing the number of images available for training.

### Key Function

- **duplicate_image(file_path, output_dir, start_index, num_copies):** Duplicates the given image a specified number of times and saves them with new filenames.

### Requirements

To run these scripts, you need to have the following Python packages installed:

- OpenCV
- NumPy
- Pandas
- PIL (Pillow)

Install the necessary packages using pip:

```bash
pip install opencv-python numpy pandas pillow
```

### Usage

Each script can be executed independently. For example, to run the `rotate.py` script, use the following command:

```bash
python rotate.py
```

Ensure the input data and output directories are correctly set within each script to match your dataset location and desired output path.

---

## Custom Deep Learning Models for 6DoF Prediction

This folder contains various custom deep learning models designed to predict the 6 Degrees of Freedom (6DoF) for a robot. These models employ different backbone architectures to facilitate robust and accurate predictions from multimodal data sources.

### vgg.ipynb

This notebook implements a custom VGG-based model to predict the 6DoF for the robot. The model architecture adapts the classic VGG design to work with multimodal inputs and outputs tailored for robotics applications.

### ResNet50_Yusuf.ipynb

This notebook features a modified ResNet50 architecture, focusing on predicting 6DoF from input data. The network is optimized to handle high-dimensional data typical in robotic perception and provides accurate pose estimation.

### ResNet50-Copy1.ipynb

This variant of ResNet50 implements minor tweaks and modifications over the standard ResNet50 to enhance performance for the specific 6DoF prediction tasks. It offers insights into how small changes can impact model performance in robotics contexts.

### ResNet50.ipynb

An implementation of the standard ResNet50 architecture tailored for 6DoF prediction. This notebook provides a baseline for comparing other architectures and enhancements in pose estimation tasks.

### ResNet18-pretrained.ipynb

This notebook uses a pretrained ResNet18 model, fine-tuned for predicting 6DoF. It leverages transfer learning to speed up training and improve performance with limited data.

### resnet.ipynb

This notebook offers another version of ResNet, possibly exploring different configurations or preprocessing steps to optimize 6DoF prediction for robotics.

### posenet-Copy1.ipynb

This notebook implements a variant of the PoseNet architecture, customized for predicting 6DoF. PoseNet is widely used in robotics for accurate pose estimation, and this version may include specific adjustments for enhanced accuracy.

### posenet.ipynb

A direct implementation of the PoseNet model for predicting the robot's 6DoF, providing robust results by utilizing pose estimation techniques suited for dynamic environments.

### custom.ipynb

This notebook presents a custom-designed neural network for 6DoF prediction, incorporating innovative layers and connections to improve prediction accuracy and model robustness.

### Requirements

To run these notebooks, ensure you have the following Python packages installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

You can install the necessary packages using pip:

```bash
pip install tensorflow keras numpy matplotlib opencv-python scikit-learn
```

### Usage

Each notebook can be run independently in a Jupyter environment. Load the notebook of interest and execute the cells to train and evaluate the model. Ensure you have the necessary dataset available and properly formatted for the training process.

---

## Scripts for Robotic Vision and Tracking

This folder contains various scripts for robotic vision and tracking tasks using OpenCV and ArUco markers. These scripts are designed to facilitate color detection, shape recognition, and ArUco marker tracking and manipulation for robotic applications.

### track_arucoMarker.py

This script tracks the movement of ArUco markers in a video stream and displays their positions. It draws the marker paths on the screen, allowing for real-time visualization of marker movement. It also integrates with a robotic system to move based on marker positions.

### shape_detection.py

This script detects geometric shapes within a video stream using contours. It identifies shapes based on the number of sides and can be used to detect and classify various shapes in real-time.

### mimic_aruco.py

This script simulates or mimics ArUco marker detection, generating virtual markers and manipulating them. It can be used for testing and development purposes without requiring physical markers.

### m.py

This script tracks ArUco markers and monitors their positions, printing messages when changes in position are detected. It is useful for observing marker behavior over time and reacting to positional changes.

### countour_arucomarker.py

This script captures video from a webcam and detects the largest contour of a specific color (red) using HSV color space. It tracks the detected contour's center and draws its path on a separate tracking frame.

### colour_detection.py

This script detects colors in a video stream, focusing on red color detection using HSV color space. It draws bounding boxes around detected areas and tracks their movement.

### aruco.py

This script detects and identifies ArUco markers in a video stream using OpenCV's ArUco module. It draws the marker boundaries and IDs on the video frame, allowing for easy identification and tracking of markers.

### Requirements

To run these scripts, you need to have the following Python packages installed:

- OpenCV
- NumPy
- Imutils
- Pandas

Install the necessary packages using pip:

```bash
pip install opencv-python numpy imutils

 pandas
```

### Usage

Each script can be executed independently. For example, to run the `track_arucoMarker.py` script, use the following command:

```bash
python track_arucoMarker.py
```

Ensure that your camera is properly configured and connected, as most scripts rely on webcam input for real-time processing.

---

## Testing and Model Evaluation Scripts

This section covers scripts designed for testing and evaluating the trained models on live or recorded data.

### streamlit_app.py

This script provides a Streamlit web application for live robot movement prediction based on video input. Users can select between live video or video file input, upload a target image, and see the predicted robot movements in real time.

### pipline1.py

This script outlines a pipeline for processing live video or video files, predicting the robot's 6DoF using the trained model, and sending the control commands to the robot. It integrates video capture, model prediction, and robot control in a streamlined process.

### Python Version Requirement

These scripts require Python 3.11.9 to ensure compatibility and optimal performance.

---

