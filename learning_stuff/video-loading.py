#!/usr/bin/env python3

import numpy as np
import cv2

cap = cv2.VideoCapture('train.mp4')
#cap = cv2.VideoCapture('train.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is True:

        # Display the resulting frame
        cv2.imshow('Frame',frame)
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
 
    # Break the loop
    else: 
        break
 
# When everything done,

'''
    ret, frame = cap.read()
    if ret == True:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

cap.release()
cv2.destroyAllWindows()
