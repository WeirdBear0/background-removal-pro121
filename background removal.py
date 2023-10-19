# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('mountain.jpg')
# resizing the mountain image as 640 X 480
newmtn = cv2.resize(mountain, (640,480))


while True:
    # read a frame from the attached camera
    status , frame = camera.read()
    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame , 1)
        # converting the image to RGB for easy processing
        frame_hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
        # creating thresholds
        lower_bound = np.array([0, 120, 50])
        upper_bound = np.array([10, 255,255])
        # thresholding image
        mask_1 = cv2.inRange(frame_hsv, lower_bound, upper_bound)
        # creating thresholds
        lower_bound = np.array([170, 120, 70])
        upper_bound = np.array([180, 255, 255])
        # thresholding image
        mask_2 = cv2.inRange(frame_hsv, lower_bound, upper_bound)
        mask_1 = mask_1 + mask_2
        cv2.imshow("mask1", mask_1)
        # inverting the mask
        mask_2 = cv2.bitwise_not(mask_1)
        # bitwise and operation to extract foreground / person
        result_1 = cv2.bitwise_and(frame, frame, mask = mask_2)
        result_2 = cv2.bitwise_and(newmtn, newmtn, mask = mask_1)
        # final image
        frame = cv2.addWeighted(result_1, 1, result_2, 1, 0)
        # show it
        cv2.imshow('frame' , frame)
        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break
# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
