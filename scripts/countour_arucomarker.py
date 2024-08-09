import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# List to store the tracked points
points = []

# Start a while loop
while True:
    # Reading the video from the webcam in image frames
    _, imageFrame = webcam.read()

    # Convert the imageFrame in BGR(RGB color space) to HSV(hue-saturation-value) color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and define mask
    red_lower = np.array([136, 50, 40], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Morphological Transform, Dilation for each color and bitwise_and operator between imageFrame and mask determines to detect only that particular color
    kernel = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)

    # Creating contour to track red color
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # A frame to display the tracking path
    track_frame = np.zeros_like(imageFrame)

    if largest_contour is not None and max_area > 300:
        # Find the center of the largest contour
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(largest_contour)
        center = (int(center_x), int(center_y))
        radius = int(radius)

        # Append the center point to the list of tracked points
        points.append(center)

        # Draw the circle around the largest red contour
        imageFrame = cv2.circle(imageFrame, center, radius, (0, 0, 255), 2)
        cv2.putText(imageFrame, "Red Colour", (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    # Draw the tracking path
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        cv2.line(track_frame, points[i - 1], points[i], (0, 255, 0), 2)


    cv2.line(imageFrame, (0, imageFrame.shape[0] // 2), (imageFrame.shape[1], imageFrame.shape[0] // 2), (0, 0, 255), 2) # x-axis (blue)
    cv2.line(imageFrame, (imageFrame.shape[1] // 2, 0), (imageFrame.shape[1] // 2, imageFrame.shape[0]), (0, 255, 0), 2) # y-axis (green)
    # Overlay the tracking path on the original frame
    combined_frame = np.hstack((imageFrame, track_frame))
    #  cv2.addWeighted(imageFrame, 0.8, track_frame, 1, 0)

    # Program Termination
    cv2.imshow("Tracking Path", combined_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
