#!/usr/bin/env python3

import numpy as np
import cv2

source_video = cv2.VideoCapture('train.mp4')
destination_video_codec = cv2.VideoWriter_fourcc(*'XVID')
destination_video = cv2.VideoWriter(
    'output.avi',
    destination_video_codec,
    20.0,
    (640, 480)
)

# Check if camera opened successfully
if (source_video.isOpened()== False): 
    print("Error opening video stream or file")
 
while(source_video.isOpened()):
    ret, source_frame = source_video.read()
    if ret is True:
        gray_frame = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        destination_video.write(gray_frame)

        # Display the resulting frame
        cv2.imshow('Source', source_frame)
        cv2.imshow('Destination', gray_frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
 
    # Break the loop
    else: 
        break
 
source_video.release()
cv2.destroyAllWindows()
