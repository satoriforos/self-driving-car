# My project's README

What is this thing?
===================
This is my feeble attempt to get python to determine the speed of a car from a dash cam video.

So far it does this:
--------------------
1. Open a video
2. change the perspective of the video to a birds-eye view of the road
3. Isolate the road lines from the background
4. Identify road lines visually

What it does not do yet:
------------------------
1. Know which road lines are which
2. Measure the distance between road lines
3. Track the speed at which the road lines are moving
4. Train the system to estimate the speed of the car from the lane lines
5. Estimate the speed of a second video


Installing Libraries
====================
```
sudo apt-get install python-opencv
pip3 install numpy
pip3 install matplotlib
pip3 install opencv-python
sudo apt-get install python3-tk
sudo pip3 install imutils
```

Running
=======
```
$ ./test_roadlinetracker.py
```


Tutorials I used:
==================

Supporting files at https://github.com/C-Aniruddh/realtime_object_recognition

Setting up OpenCV:
https://pythonprogramming.net/loading-images-python-opencv-tutorial/

How to import OpenCV in Python: https://stackoverflow.com/questions/46610689/how-to-import-cv2-in-python3

Loading images: https://pythonprogramming.net/loading-images-python-opencv-tutorial/

Image operations: https://pythonprogramming.net/image-operations-python-opencv-tutorial/

Reading and Writing video: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

Haar Cascade Object Detection: https://pythonprogramming.net/haar-cascade-object-detection-python-opencv-tutorial/

Calculate average velocity: https://stackoverflow.com/questions/32729920/how-can-i-calculate-the-average-speed

Image Thresholding: https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html

Shape detection: https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/

Perspective Transformation: http://zqdevres.qiniucdn.com/data/20170602165702/index.html

YOLO Object Detection: https://www.youtube.com/watch?v=4eIBisqx9_g

Real-time speed estimation of moving objects: http://www.amphioxus.org/content/real-time-speed-estimation-cars

Road Sign Recognition: https://github.com/nikgens/TankRobotProject/tree/master/signRecognition

Custom Object Detection: http://blog.dlib.net/2014/02/dlib-186-released-make-your-own-object.html

Detect which lane a car is in: https://github.com/navoshta/detecting-road-features

Motion capture: https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

Detect birds eating at a bird feeder: https://www.makeartwithpython.com/blog/poor-mans-deep-learning-camera/
