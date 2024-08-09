# Stewart-Platform-
This repository contains code for training custom multi modal neural networks designed to predict the 6 Degrees of Freedom (6DoF) for parallel robot.
Here is a README for your GitHub repository's `scripts` folder based on the contents of the uploaded files:

---

## Scripts for Robotic Vision and Tracking

This folder contains various scripts for robotic vision and tracking tasks using OpenCV and ArUco markers. These scripts are designed to facilitate color detection, shape recognition, and ArUco marker tracking and manipulation for robotic applications.

### Table of Contents

- [track_arucoMarker.py](#track_arucomarkerpy)
- [shape_detection.py](#shape_detectionpy)
- [mimic_aruco.py](#mimic_arucopy)
- [m.py](#mpy)
- [countour_arucomarker.py](#countour_arucomarkerpy)
- [colour_detection.py](#colour_detectionpy)
- [aruco.py](#arucopy)
- [Requirements](#requirements)
- [Usage](#usage)

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

## colour_detection.py

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
pip install opencv-python numpy imutils pandas
```

### Usage

Each script can be executed independently. For example, to run the `track_arucoMarker.py` script, use the following command:

```bash
python track_arucoMarker.py
```

Ensure that your camera is properly configured and connected, as most scripts rely on webcam input for real-time processing.

--- 
