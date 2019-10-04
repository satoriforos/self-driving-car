#!/usr/bin/env python3

# https://github.com/colinmccormick/CarND-Advanced-Lane-Lines

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
#from IPython import get_ipython

# set parameters

#get_ipython().run_line_magic('matplotlib', 'inline')

# processing mode
TEST_MODE = False
RECORD = False

# image size
ROWS = 720
COLS = 1280
ROI_TOP = 0.65
ROI_BOTTOM = 0.90
ROI_LEFT = 0.10
ROI_RIGHT = 0.95
INSET_ROWS = int(0.4 * ROWS)
INSET_COLS = int(0.4 * COLS)

# colors and scales
RED_COLOR = (0,0,255)
GREEN_COLOR = (0,255,0)
XM_PER_PIX = 3.7 / 700
YM_PER_PIX = 15 / 720   # based on estimate of 5 white lane dashes fitting vertically in warped image; 3m each

# camera calibration
CHESS_NX = 9
CHESS_NY = 6

# lane pixel finding
L_THRESHOLD = [225,255]              # range for L channel in LUV color space (for white)
B_THRESHOLD = [190,255]              # range for B channel in LAB color space (for yellow)

# perspective transform
GB_KERNEL = 3                       # size of gaussian blur kernel
CANNY_LOW = 50                      # canny edge detection threshold (low)
CANNY_HIGH = 400                    # canny edge detection threshold (high)
HOUGH_DIST_RES = 0.5                # hough line finder distance resolution (pixels)
HOUGH_ANGLE_RES = 3.14159/180       # hough line finder angle resolution (rads)
HOUGH_THRESHOLD = 20                # hough line finder threshold
HOUGH_MIN_LINE = 60                 # hough line finder min line length (pixels)
HOUGH_MAX_GAP = 120                 # hough line finder max line gap (pixels)

# lane fitting
NUM_WINS = 8                        # number of sliding windows to use
WIN_WIDTH = 64                      # half total width of sliding window
WIN_HEIGHT = int(ROWS/NUM_WINS)     # total height of sliding window
FRAME_MEMORY = 4                    # number of past frames to remember
LANE_WIDTH_PIXELS = 810             # width of lane at bottom of image in pixels
FRAME_MARKER_THRESHOLD = 4          # frames to wait until new lane marker can reasonably appear 

# speed measurement
METERS_PER_MARKER = 15.75           # meters between white lane markers (40 feet)

# directories and filenames
TEST_DIR = 'test_images'
CALIBRATION_DIR = 'camera_cal'
CALIBRATION_FILENAME = 'camera_calibration.pickle'
PERSPECTIVE_FILENAME = 'perspective_transform.pickle'

OUTPUT_DIR_CALIBRATION = 'output_images/calibration'
OUTPUT_DIR_S_CHANNEL = 'output_images/s_channel'
OUTPUT_DIR_X_GRAD = 'output_images/x_gradient'
OUTPUT_DIR_VP = 'output_images/vanishing_point'

OUTPUT_DIR_UNDISTORT = 'output_images/undistort'
OUTPUT_DIR_WARPED = 'output_images/warped'
OUTPUT_DIR_MARKER_PIXELS = 'output_images/marker_pixels'
OUTPUT_DIR_SLIDING_WINDOW = 'output_images/sliding_window'
OUTPUT_DIR_LANE_MARK = 'output_images/lane_mark'

class LaneLine(object):
    # class to hold info about each lane line

    def __init__(self,side):
        # side: 'left' or 'right'
        self.side = side
        self.goodFitLastFrame = False
        self.recentFits = np.zeros((FRAME_MEMORY,3),dtype=np.float32)      # second-order polynomial: 3 coefficients
        self.curvatureRadius = np.zeros((FRAME_MEMORY,),dtype=np.float32)
        self.framesSeenSinceMarker = 0
        self.estimatedSpeed = 0

    def calculateCurvature(self,f):
        # calculate radius of curvature; use some algebra to scale pixels to meters

        y0 = ROWS * YM_PER_PIX
        A = f[0] *  XM_PER_PIX / YM_PER_PIX**2
        B = f[1] * XM_PER_PIX / YM_PER_PIX
            
        curvature = ((1 + (2*A*y0 + B)**2)**1.5)/float(np.absolute(2*A))
        self.curvatureRadius = np.roll(self.curvatureRadius,1)
        self.curvatureRadius[0] = curvature
          
        return np.mean(self.curvatureRadius)
        # define camera calibration

def calibrateCamera():
    # use chessboard photos to calibrate camera

    # set termination criteria (for use with cv2.cornerSubPix)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # make object points
    objp = np.zeros((CHESS_NX*CHESS_NY,3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESS_NX,0:CHESS_NY].T.reshape(-1,2)
    imagePoints = []
    objectPoints = []

    # loop through calibration images
    for imageFile in os.listdir(CALIBRATION_DIR):
        if imageFile.endswith('.jpg'):

            # read image, grayscale, find corners
            im = cv2.imread(os.path.join(CALIBRATION_DIR,imageFile))
            gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(gray,(CHESS_NX,CHESS_NY),None)

            if found:
                # draw corners on image
                cv2.drawChessboardCorners(im,(CHESS_NX,CHESS_NY),corners,found)
                # find corners with higher (sub-pixel) precision
                corners2 = cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
                imagePoints.append(corners2)
                objectPoints.append(objp)

                # save image with corners marked
                cv2.imwrite(os.path.join(OUTPUT_DIR_CALIBRATION,imageFile),im)

    # do the actual calibration and save
    ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(objectPoints,imagePoints,gray.shape[::-1],None,None)
    imSize = im.shape
    calibrationData = {'cam_matrix':mtx, 'dist_coeffs':dst, 'im_size':imSize}
    with open(CALIBRATION_FILENAME,'wb') as f:
        pickle.dump(calibrationData,f)

    return mtx,dst

# do camera calibration and display example images

print("Calibrating camera...")
mtx,dst = calibrateCamera()
filenameList = [fn for fn in os.listdir(CALIBRATION_DIR) if fn.endswith('.jpg')]
fn = random.choice(filenameList)
imOriginal = cv2.imread(os.path.join(CALIBRATION_DIR,fn))
imCorners = cv2.imread(os.path.join(OUTPUT_DIR_CALIBRATION,fn))

f = plt.figure(figsize=(30,40))
ax = f.add_subplot(1,2,1)
ax.imshow(imOriginal)
ax = f.add_subplot(1,2,2)
ax.imshow(imCorners)
plt.show()

# display undistorted image example

print("Undistorting image...")
fn = os.path.join(CALIBRATION_DIR,'calibration1.jpg')
imCB = cv2.imread(fn)
imCB = cv2.cvtColor(imCB,cv2.COLOR_BGR2RGB)
imCBUndistorted = cv2.undistort(imCB,mtx,dst)
f2 = plt.figure(figsize=(30,40))
ax = f2.add_subplot(1,2,1)
ax.imshow(imCB)
ax = f2.add_subplot(1,2,2)
ax.imshow(imCBUndistorted)
plt.show()

filenameList = [fn for fn in os.listdir(TEST_DIR) if fn.endswith('.jpg')]
fn = random.choice(filenameList)
imOriginal = cv2.imread(os.path.join(TEST_DIR,fn))
imOriginalRGB = cv2.cvtColor(imOriginal,cv2.COLOR_BGR2RGB)
imUndistorted = cv2.undistort(imOriginalRGB,mtx,dst)
f3 = plt.figure(figsize=(30,40))
ax = f3.add_subplot(1,2,1)
ax.imshow(imOriginalRGB)
ax = f3.add_subplot(1,2,2)
ax.imshow(imUndistorted)
plt.show()

# define lane pixel finder

def selectLanePixels(im, fn='frame.jpg'):
    # select pixels that are lane markers; assume RGB image; use LUV, LAB color spaces

    luv = cv2.cvtColor(im,cv2.COLOR_RGB2LUV)
    lab = cv2.cvtColor(im,cv2.COLOR_RGB2Lab)

    # find yellow lane pixels using LUV color space
    lChannel = luv[:,:,0]
    lBinary = np.zeros_like(lChannel)
    lBinary[ (lChannel > L_THRESHOLD[0]) & (lChannel <= L_THRESHOLD[1]) ] = 255
    
    # find white lane pixels using LAB color space
    # treat lab b-channel differently: normalize up to 255
    # however, skip this if max value is low (implying no yellow in the image)
    # thanks to jeremy shannon for this idea: https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines
    bChannel = lab[:,:,2]
    bMax = np.max(bChannel)
    if bMax > 175:
        bChannel = 255 * ( bChannel / bMax )
    bBinary = np.zeros_like(bChannel)
    bBinary[ (bChannel > B_THRESHOLD[0]) & (bChannel <= B_THRESHOLD[1]) ] = 255

    # make combined mask
    lanePixels = np.zeros_like(lChannel)
    lanePixels[ (lBinary > 0) | (bBinary > 0) ] = 255

    return lanePixels, lBinary, bBinary


# display example lane pixel finder

print("Finding example lane pixels...")

filenameList = [fn for fn in os.listdir(TEST_DIR) if fn.endswith('.jpg')]
fn = random.choice(filenameList)
imBGR = cv2.imread(os.path.join(TEST_DIR,fn))
imRGB = cv2.cvtColor(imBGR,cv2.COLOR_BGR2RGB)

lanePixels, lBinary, bBinary = selectLanePixels(imRGB,fn)

f = plt.figure(figsize=(30,40))
ax = f.add_subplot(1,2,1)
ax.imshow(imRGB)
ax = f.add_subplot(1,2,2)
ax.imshow(lanePixels,cmap='gray')
plt.show()

# define utility function for selecting roi

def selectRoi(im):
    # mask out any pixels outside of the roi

    #rows,cols = im.shape[:2]
    bottomLeft = [0,ROWS*ROI_BOTTOM]
    topLeft = [COLS*ROI_LEFT,ROWS*ROI_TOP]
    topRight = [COLS*ROI_RIGHT,ROWS*ROI_TOP]
    bottomRight = [COLS,ROWS*ROI_BOTTOM]
    vertices = np.array([[bottomLeft,topLeft,topRight,bottomRight]],dtype=np.int32)
    mask = np.zeros_like(im)
    cv2.fillPoly(mask,vertices,255)
    
    return cv2.bitwise_and(im,mask)

# define utility function for determining line intersection

def findIntersection(lines):
    # find the point minimizing lsq distance from all lines
    # if the lines all intersect, this is the intersection point

    numLines = len(lines)
    a = np.zeros((numLines,2),dtype=np.float32)
    b = np.zeros((numLines,),dtype=np.float32)
    for n,line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            slope = (y2-y1) / float(x2-x1)
            a[n] = np.array([slope,-1],dtype=np.float32)
            b[n] = slope * x1 - y1     # this is -1 times the intercept
    return np.linalg.lstsq(a,b)[0]

# define utility function for finding vanishing point

def findVanishingPoint(mtx,dst):
    # determine vanishing point in image

    # analyze straight-road images used to find vanishing point
    straightImages = [os.path.join(TEST_DIR,f) for f in os.listdir(TEST_DIR) if 'straight' in f]
    vps = []
    for imFile in straightImages:
        im = cv2.imread(imFile)
        im = cv2.undistort(im,mtx,dst)

        # smooth with gaussian blur
        imSmoothed = cv2.GaussianBlur(im,(GB_KERNEL,GB_KERNEL),0)

        # find edges with canny
        edges = cv2.Canny(imSmoothed,CANNY_LOW,CANNY_HIGH)

        # apply roi mask
        edgesRoi = selectRoi(edges)
        
        # find lines with hough; use probabilistic version to increase speed
        lines = cv2.HoughLinesP(edgesRoi,HOUGH_DIST_RES,HOUGH_ANGLE_RES,HOUGH_THRESHOLD,None,HOUGH_MIN_LINE,HOUGH_MAX_GAP)
        
        # identify best overlap point of lines - this is the vanishing point
        vp = findIntersection(lines)
        vps.append(vp)

        # save image
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(im, (x1,y1), (x2,y2), GREEN_COLOR, thickness=2) 
        cv2.circle(im,(vp[0],vp[1]),8,GREEN_COLOR,2)
        savepath = os.path.join(OUTPUT_DIR_VP,os.path.split(imFile)[1])
        cv2.imwrite(savepath,im)

    # return arithmetic mean of vp coordinates
    return np.mean(vps,axis=0)


# display example of vanishing point

print("Calculating vanishing point...")
vp = findVanishingPoint(mtx,dst)
print("Vanishing point: {0}".format(vp))
filenameList = [fn for fn in os.listdir(OUTPUT_DIR_VP) if fn.endswith('.jpg')]
fn = random.choice(filenameList)
imBGR = cv2.imread(os.path.join(OUTPUT_DIR_VP,fn))
imRGB = cv2.cvtColor(imBGR,cv2.COLOR_BGR2RGB)
plt.imshow(imRGB)
plt.show()

# define function for calculating perspective transform

def calculatePerspectiveTransform():
    # calculate perspective transform matrix

    # read in camera calibration
    with open(CALIBRATION_FILENAME,'rb') as f:
        calibrationData = pickle.load(f)
    mtx = calibrationData['cam_matrix']
    dst = calibrationData['dist_coeffs']

    # find vanishing point; note this is returned as (x,y)
    vp = findVanishingPoint(mtx,dst)
        
    xVP,yVP = int(vp[0]),int(vp[1])
    xBottomLeft = int(ROI_LEFT * COLS)
    xBottomRight = int(ROI_RIGHT * COLS)
    yTop = int(ROWS * ROI_TOP)
    yBottom = int(ROWS * ROI_BOTTOM)
    
    # calculate x positions of upper ROI corners
    # x = my + b; m = (x1-x0)/(y1-y0); b = x0 - m * y0
    leftM = (xVP - xBottomLeft) / float(yVP - yBottom)    
    leftB = xBottomLeft - leftM * yBottom
    rightM = (xVP - xBottomRight) / float(yVP - yBottom)
    rightB = xBottomRight - rightM * yBottom
    xTopLeft = int(leftM * yTop + leftB)
    xTopRight = int(rightM * yTop + rightB)
            
    corners = [(xTopLeft,yTop),(xBottomLeft,yBottom),(xBottomRight,yBottom),(xTopRight,yTop)]
    pts1 = np.float32(corners)  
    pts2 = np.float32( [[0,0],[0,ROWS],[COLS,ROWS],[COLS,0]] )
    
    perspectiveTrans = cv2.getPerspectiveTransform(pts1,pts2)
    inversePerspectiveTrans = cv2.getPerspectiveTransform(pts2,pts1)

    transformDict = {'forwardWarp': perspectiveTrans, 'backwardWarp': inversePerspectiveTrans,'corners': corners}

    with open(PERSPECTIVE_FILENAME,'wb') as f:
        pickle.dump(transformDict,f)

    return transformDict

    # display example perspective warp

print("Warping perspective forward and backward...")

td = calculatePerspectiveTransform()
p = td['corners']

filenameList = [fn for fn in os.listdir(TEST_DIR) if fn.endswith('.jpg') and 'straight' in fn]
fn = random.choice(filenameList)
im = cv2.imread(os.path.join(TEST_DIR,fn))
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
im = cv2.undistort(im,mtx,dst)

cv2.line(im,p[0],p[1],color=(255,0,0),thickness=4)
cv2.line(im,p[1],p[2],color=(255,0,0),thickness=4)
cv2.line(im,p[2],p[3],color=(255,0,0),thickness=4)
cv2.line(im,p[3],p[0],color=(255,0,0),thickness=4)

imWarped = cv2.warpPerspective(im,td['forwardWarp'],(COLS,ROWS))
imWarpedBack = cv2.warpPerspective(imWarped,td['backwardWarp'],(COLS,ROWS))

f = plt.figure(figsize=(30,40))
ax = f.add_subplot(1,3,1)
ax.imshow(im)
ax = f.add_subplot(1,3,2)
ax.imshow(imWarped)
ax = f.add_subplot(1,3,3)
ax.imshow(imWarpedBack)
plt.show()

def findCentroid(im):
    # utility function to find horizontal centroid of distribution

    hist = np.sum(im,axis=0)
    nonzero = np.flatnonzero(hist)
    if len(nonzero) > 0:
        centroid = int(nonzero.mean())
        found = True
    else:
        centroid = int(0.5 * im.shape[1])
        found = False
 
    return found, centroid

    # define speed estimation routine
# watches for lane markers to pass and counts frames since last one

def estimateSpeed(roi):
    # check for lane marker
    hist = np.sum(roi,axis=1)
    nonzero = np.flatnonzero(hist)
    if len(nonzero) == len(hist):
        if rightLane.framesSeenSinceMarker > FRAME_MARKER_THRESHOLD:
            # we found a new lane marker
            metersTraveled = METERS_PER_MARKER
            secondsTaken = rightLane.framesSeenSinceMarker / float(FPS)
            rightLane.estimatedSpeed = (metersTraveled / secondsTaken) * (0.001) * 3600
            rightLane.framesSeenSinceMarker = 0

            # define main routine to fit lane lines for left and right

def findAndFit(im,yVals):

    # set up image for drawing
    imFit = np.copy(im)
    imFit = cv2.cvtColor(imFit,cv2.COLOR_GRAY2BGR)
    proposedFits = []
    
    # increment frame count
    rightLane.framesSeenSinceMarker += 1
    
    for lane in ([leftLane,rightLane]):
        xCentroids = [] 
        if lane.goodFitLastFrame:     
            # case where we got a good fit last frame
            fit = np.poly1d(lane.recentFits[0])
            for i in range(len(yVals)):
                y = yVals[i]
                currentXValue = int(fit(y))
                xLeft,xRight = currentXValue - WIN_WIDTH, currentXValue + WIN_WIDTH
                yBottom,yTop = y, y - WIN_HEIGHT
                roi = im[yTop:yBottom,xLeft:xRight]
                found, centroid = findCentroid(roi)
                xCentroids.append(centroid + xLeft)
                cv2.rectangle(imFit,(xLeft,yTop),(xRight,yBottom),GREEN_COLOR,2)
                imFit[yTop:yBottom,currentXValue] = RED_COLOR
                
                # do speed estimation
                if (i == 0) & (lane == rightLane):
                    estimateSpeed(roi)
                
        else:
            # case where we didn't get a good fit last frame
            # start by finding x position of lane at bottom of image       
            if lane.side == 'left':
                xLeft,xRight = 0,int(np.floor(COLS/2))
            elif lane.side == 'right':
                xLeft,xRight = int(np.ceil(COLS/2)),COLS
            roi = im[int(ROWS/2):ROWS,xLeft:xRight]
            found, centroid = findCentroid(roi)
            if found:
                currentXValue = centroid + xLeft
            else:
                # worst case: can't find it, so go with mid-point
                currentXValue = int(xLeft + 0.25 * ROWS)
            for i in range(len(yVals)):
                y = yVals[i]
                xLeft, xRight = currentXValue - WIN_WIDTH, currentXValue + WIN_WIDTH
                yBottom, yTop = y, y-WIN_HEIGHT
                roi = im[yTop:yBottom,xLeft:xRight]
                found, centroid = findCentroid(roi)
                if found:
                    currentXValue = centroid + xLeft
                xCentroids.append(currentXValue)
                cv2.rectangle(imFit,(xLeft,yTop),(xRight,yBottom),GREEN_COLOR,2)
                imFit[yTop:yBottom,currentXValue] = RED_COLOR

        # do polynomial fit
        xCentroidsArray = np.array(xCentroids)
        proposedFit = np.polyfit(yVals,xCentroidsArray,2)     # 2nd-order polynomial fit            
        proposedFits.append(proposedFit)
            
    # do sanity check    
    fitLeft = np.poly1d(proposedFits[0])
    fitRight = np.poly1d(proposedFits[1])
    xLeft = fitLeft(ROWS)
    xRight = fitRight(ROWS)
    if (np.absolute(xRight - xLeft - LANE_WIDTH_PIXELS) < WIN_WIDTH):
        leftLane.goodFitLastFrame = True
        rightLane.goodFitLastFrame = True
        leftLane.recentFits = np.roll(leftLane.recentFits,1,axis=0)
        rightLane.recentFits = np.roll(rightLane.recentFits,1,axis=0)
        leftLane.recentFits[0] = proposedFits[0]
        rightLane.recentFits[0] = proposedFits[1]
 
    return imFit

    # define function to draw lane on image

def drawLane(im):

    # get pixels in lane from fits (in warped space)
    yPixelVals = range(ROWS)
    
    if TEST_MODE:
        # don't average
        leftFit = leftLane.recentFits[0]
        rightFit = rightLane.recentFits[0]
    else:
        # video mode; do average
        leftFit = np.mean(leftLane.recentFits,axis=0)
        rightFit = np.mean(rightLane.recentFits,axis=0)
        
    fitLeft = np.poly1d(leftFit)
    fitRight = np.poly1d(rightFit)
    xPixelValsLeft = fitLeft(yPixelVals)
    xPixelValsRight = fitRight(yPixelVals)

    pts = zip(yPixelVals,xPixelValsLeft,xPixelValsRight)

    # draw lane on warped image
    imLaneWarped = np.zeros((ROWS,COLS,3),np.uint8)
    for p in pts:
        imLaneWarped[p[0],int(p[1]):int(p[2]),:] = GREEN_COLOR

    # warp image back and combine with original image
    imLane = cv2.warpPerspective(imLaneWarped,inversePerspectiveTrans,(COLS,ROWS))    
    imLane = cv2.addWeighted(im,1,imLane,0.5,0)

    # calculate curvature
    curveLeft = leftLane.calculateCurvature(leftFit)
    curveRight = rightLane.calculateCurvature(rightFit)
    curveMean = 0.5 * (curveLeft + curveRight)
    curveStr = 'Lane curvature: {:06.2f} m'.format(curveMean)
    cv2.putText(imLane,curveStr,(100,90),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),thickness=2)

    # calculate lane center
    centerOffset = 0.5*(COLS-fitLeft(ROWS)-fitRight(ROWS)) * XM_PER_PIX
    centerStr = 'Lane offset: {:3.2f} m'.format(centerOffset)
    cv2.putText(imLane,centerStr,(100,140),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),thickness=2)

    # write estimated speed on image
    speedStr = 'Est. speed: {:3.2f} km/h'.format(rightLane.estimatedSpeed)
    cv2.putText(imLane,speedStr,(100,190),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),thickness=2)

    return imLane

    # define main routine for lane finding

def findLane(im):

    # read file and undistort
    imUndistorted = cv2.undistort(im,mtx,dst)

    # find lane pixels
    imBinary, lBinary, bBinary = selectLanePixels(imUndistorted)

    # transform perspective
    imWarped = cv2.warpPerspective(imBinary,perspectiveTrans,(COLS,ROWS))

    # find lanes and fit to polynomial    
    yVals = range(ROWS,0,-WIN_HEIGHT) 
    imFit = findAndFit(imWarped,yVals)

    # draw lane on image
    imLane = drawLane(imUndistorted)    
    
    # write sliding window inset onto image
    imFitSmall = cv2.resize(imFit,(INSET_COLS,INSET_ROWS))
    imLane[32:32+INSET_ROWS,COLS-INSET_COLS-1:-1] = imFitSmall
    
    if RECORD:
        # save images at various processing stages
        cv2.imwrite(os.path.join(OUTPUT_DIR_UNDISTORT,imageFile),imUndistorted)
        cv2.imwrite(os.path.join(OUTPUT_DIR_MARKER_PIXELS,imageFile),imBinary)
        cv2.imwrite(os.path.join(OUTPUT_DIR_WARPED,imageFile),imWarped)
        cv2.imwrite(os.path.join(OUTPUT_DIR_SLIDING_WINDOW,imageFile),imFit)
        cv2.imwrite(os.path.join(OUTPUT_DIR_LANE_MARK,imageFile),imLane)

    return imLane

    # do video processing

# read in camera calibration or perform calibration if pickled file is unavailable
try:
    with open(CALIBRATION_FILENAME,'rb') as f:
        calibrationData = pickle.load(f)
        mtx = calibrationData['cam_matrix']
        dst = calibrationData['dist_coeffs']
        print("Read camera calibration from file...")
except:
    print("Performing camera calibration...")
    mtx,dst = calibrateCamera()

# read in perspective transform or perform transform matrix calculation if pickled file is unavailable
try:
    with open(PERSPECTIVE_FILENAME,'rb') as f:
        transformDict = pickle.load(f)
        perspectiveTrans = transformDict['forwardWarp']
        inversePerspectiveTrans = transformDict['backwardWarp']
    print("Read perspective transform from file...")
except:
    print("Performing perspective transform...")
    transformDict = calculatePerspectiveTransform()
    perspectiveTrans = transformDict['forwardWarp']
    inversePerspectiveTrans = transformDict['backwardWarp']

# set up lane line objects
leftLane = LaneLine('left')
rightLane = LaneLine('right')

# if we're in test mode, analyze test images
if TEST_MODE:
    for imageFile in os.listdir(TEST_DIR):
        if imageFile.endswith('.jpg') and 'straight' not in imageFile:
            print('Loading image {0}...'.format(imageFile))
            fn = os.path.join(TEST_DIR,imageFile)
            im = cv2.imread(fn)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            findLane(im)

else:
    # we're in video mode
    video_output = 'output_video/video_output.mp4'
    clip = VideoFileClip('./project_video.mp4')
    FPS = clip.fps
    clip_out = clip.fl_image(findLane)
    clip_out.write_videofile(video_output, audio=False)
