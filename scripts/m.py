import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
import time

def monitor_aruco_positions(position_stream, interval=0.0008900165557861328):
    """
    Monitor ArUco marker positions and print messages based on changes in position values.
    
    Args:
    - position_stream: a list of [x, y] positions of the ArUco marker.
    - interval: time interval in seconds to check for changes.
    """
    previous_position = None
    start_time = time.time()
    print(position_stream)

    for position in position_stream:
        current_time = time.time()
        if current_time - start_time >= interval:
            # print(current_time - start_time>=interval)
            if previous_position is not None:
                x_changed = position[0] != previous_position[0]
                y_changed = position[1] != previous_position[1]

                if x_changed and y_changed:
                    print("Both x and y are changing by "+ position[0]- previous_position[0]+"and"+ position[1]- previous_position[1] )
                elif x_changed:
                    print("x is changing by "+ position[0]- previous_position[0])
                elif y_changed:
                    print("y is changing by "+ position[1]- previous_position[1] )

            previous_position = position
            start_time = current_time

calib_data_path = "/home/yusuf/Desktop/stewart/calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coeff = calib_data["distCoef"]
r_vector = calib_data["rVector"]
t_vector = calib_data["tVector"]

MARKER_SIZE = 26  # mm
marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
param_markers = aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

my_ids = [1, 2, 3, 4]
ts_l = []
x1_l = []
y1_l = []
z1_l = []
x2_l = []
y2_l = []
z2_l = []
x3_l = []
y3_l = []
z3_l = []
x4_l = []
y4_l = []
z4_l = []
start_time = pd.Timestamp.now()

# Initialize a dictionary to store the paths of the markers
marker_paths = {1: [], 2: [], 3: [], 4: []}

# Create a blank image for tracking the paths
track_frame = np.zeros((480, 640, 3), dtype=np.uint8)

prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected.")
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coeff
        )
        detected_ids = []
        for i in range(0, len(marker_IDs)):
            detected_ids.append(marker_IDs[i][0])
        detected_ids.sort()
        detected_ids = [*set(detected_ids)]

        detected_id_index = []
        for j in detected_ids:
            index = np.where(marker_IDs == j)
            detected_id_index.append(index[0][0])

        ts = pd.Timestamp.now()
        ts_l.append(ts)
        for id in my_ids:
            if id in detected_ids:
                ind = np.where(marker_IDs == id)
                index = ind[0][0]
                x = tVec[index][0][0]
                y = tVec[index][0][1]
                z = tVec[index][0][2]
                if id == 1:
                    x1_l.append(x)
                    y1_l.append(y)
                    z1_l.append(z)
                elif id == 2:
                    x2_l.append(x)
                    y2_l.append(y)
                    z2_l.append(z)
                elif id == 3:
                    x3_l.append(x)
                    y3_l.append(y)
                    z3_l.append(z)
                elif id == 4:
                    x4_l.append(x)
                    y4_l.append(y)
                    z4_l.append(z)
                marker_paths[id].append([x, y])  # Ensure positions are lists
            else:
                if id == 1:
                    x1_l.append('NA')
                    y1_l.append('NA')
                    z1_l.append('NA')
                elif id == 2:
                    x2_l.append('NA')
                    y2_l.append('NA')
                    z2_l.append('NA')
                elif id == 3:
                    x3_l.append('NA')
                    y3_l.append('NA')
                    z3_l.append('NA')
                elif id == 4:
                    x4_l.append('NA')
                    y4_l.append('NA')
                    z4_l.append('NA')

        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[1].ravel()

            cv2.drawFrameAxes(frame, cam_mat, dist_coeff, rVec[i], tVec[i], 10, 2)

            cv2.putText(
                frame,
                f"ID: {ids[0]} X_t:{round(tVec[i][0][0])} Y_t:{round(tVec[i][0][1])} Z_t:{round(tVec[i][0][2])}",
                top_right,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    # Draw paths on the track_frame
    for id, path in marker_paths.items():
        if len(path) > 1:
            for j in range(1, len(path)):
                cv2.line(track_frame, 
                         (int(path[j-1][0] + frame.shape[1] // 2), int(path[j-1][1] + frame.shape[0] // 2)),
                         (int(path[j][0] + frame.shape[1] // 2), int(path[j][1] + frame.shape[0] // 2)), 
                         (0, 255, 0), 2)

    cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (0, 0, 255), 2) # x-axis (blue)
    cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (0, 255, 0), 2) # y-axis (green)
    
    if prev_frame is not None:
        merged_frame = np.hstack((prev_frame, frame))
        cv2.imshow("Merged Frames", merged_frame)

    prev_frame = track_frame
    # x= time.time()
    # Periodically check marker positions
    # if x - start_time >= 5.0:  # Every 5 seconds
    for id, path in marker_paths.items():
            # print(path.__sizeof__)
            monitor_aruco_positions(path, interval=0.0008900165557861328)
        # start_time = time.time()

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
