import cv2
import numpy as np

from sklearn.cluster import DBSCAN
from highlight_box import highlight_box

def draw_clusters_with_boxes(frame, points, eps=20, min_samples=3):
    
    # Reshape points to (N, 2)
    pts = points.reshape(-1, 2)
    
    

    # Cluster points
    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    except:
        return frame
    labels = clustering.labels_

    for label in set(labels):
        if label == -1:
            
            continue

        cluster_pts = pts[labels == label]
        x, y, w, h = cv2.boundingRect(cluster_pts.astype(np.int32))
        cv2.circle(frame, (int(x), int(y)), 3, [0, 255, 0], -1)
        
    for i in range(len(labels)):
        if labels[i] == -1:
            x, y = pts[i]
            cv2.circle(frame, (int(x), int(y)), 3, [0, 255, 0], -1)

    return frame

# Open the video file or stream
#cap = cv2.VideoCapture("videos/alabama_clemson.mp4")
cap = cv2.VideoCapture("videos/soccer_highlight_1_clip.mp4")

# Feature detection parameters
feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=20)

# LK optical flow parameters
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

# Read the first frame and initialize
ret, old_frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Define codec and output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or 'avc1'
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

old_vis = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (track points)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Filter out good points
    if p1 is None:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()
        continue

    good_old = p0[st == 1]
    good_new = p1[st == 1]

    # Estimate homography
    if len(good_old) >= 4:
        H, mask_h = cv2.findHomography(good_old, good_new, cv2.RANSAC, 3.0)
        inliers = mask_h.ravel() == 1
    else:
        inliers = np.ones(len(good_old), dtype=bool)

    # Draw tracked points
    vis = frame.copy()
    
    outliers = []
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        # green = background, red = object
        if not inliers[i]:
            outliers.append([a, b])

    vis = draw_clusters_with_boxes(vis, np.array(outliers))
    
    

    cv2.imshow("Camera vs Object Motion", vis)

    # Break with ESC
    if cv2.waitKey(1) == 27:
        break

    # Update for next frame
    old_gray = frame_gray.copy()
    
    
    if len(good_new.reshape(-1, 1, 2)) > 50:
        p0 = good_new.reshape(-1, 1, 2)

    # Re-detect if too few points
    '''if len(p0) < 50:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)'''
    
    out.write(vis)

cap.release()
out.release()
cv2.destroyAllWindows()
