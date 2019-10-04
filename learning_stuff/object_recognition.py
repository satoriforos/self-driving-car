#!/usr/bin/env python3

import numpy as np
import cv2

args = {
    "confidence": 0.1,
    'model': 'MobileNetSSD_deploy.caffemodel',
    'prototxt': 'MobileNetSSD_deploy.prototxt.txt',
}

source_video = cv2.VideoCapture('train.mp4')
destination_video_codec = cv2.VideoWriter_fourcc(*'XVID')
destination_video = cv2.VideoWriter(
    'output.avi',
    destination_video_codec,
    20.0,
    (640, 480)
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Check if camera opened successfully
if (source_video.isOpened()== False): 
    print("Error opening video stream or file")
 
while(source_video.isOpened()):
    ret, source_frame = source_video.read()
    if ret is True:
        gray_frame = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        destination_video.write(gray_frame)

        (h, w) = source_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(source_frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )
        print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
         
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
         
                # display the prediction
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                print("[INFO] {}".format(label))
                cv2.rectangle(source_frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(source_frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

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
