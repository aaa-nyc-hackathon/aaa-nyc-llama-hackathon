import numpy as np
import cv2 as cv
import argparse
import os
import math
from clip_video import clip_video
from highlight_box import highlight_box
'''parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')'''
cap = cv.VideoCapture(os.path.join(os.getcwd(), "videos/soccer_highlight_1 _clip.mp4"))
#cap = cv.VideoCapture(os.path.join(os.getcwd(), "output.mp4"))
#clip_video(cap) # Clip bottom of video to remove overlay
#cap = cv.VideoCapture(os.path.join(os.getcwd(), "videos/tmp_clip.mp4"))
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.3,
                       minDistance = 40,
                       blockSize = 15 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 20,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.05))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# Fixed player target box
player_box_width = 30
player_box_height = 60

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    # Find contours from the edge image
    frame_gray = cv.Canny(frame_gray, threshold1=50, threshold2=200, apertureSize=7)
    contours, _ = cv.findContours(frame_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create a blank canvas to draw filled shapes
    filled = np.zeros_like(frame)  # Optional: color background
    
    # Draw and fill each contour
    shapes = cv.drawContours(filled, contours, -1, (0, 255, 0), thickness=cv.FILLED)
    
    gray = cv.cvtColor(shapes, cv.COLOR_BGR2GRAY)
    
    # Threshold to get binary mask
    _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    
    intersection = cv.bitwise_xor(binary, frame_gray)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        print(math.sqrt((a-c)**2+(b-d)**2) < 1e-2)
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        frame = highlight_box(frame, a, b)
        
        
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()