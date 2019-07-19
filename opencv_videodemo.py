import cv2
import numpy as np

"""
This code main achieve the function:
Set a mask for the blue object that appears in the captured video
"""

cap = cv2.VideoCapture(0)  # Capture the video

while():
    # 1. get one frame
    ret, frame = cap.read()

    # 2.Color space convert to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # 3. set a threshold for blue
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # 4.produce a mask depend on the threshold for blue
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 5.bit operation for mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # 6. show the video
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# 7. close all windows
cv2.destroyAllWindows()