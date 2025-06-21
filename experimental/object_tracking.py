import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2
from segmentation import segment_court_frames
from typing import Generator
import numpy as np
import random
import os
from config import TMP_DIRNAME_VIDEOS, HF_MODEL
from utils import cache_video, load_frames, reduce_player_occlusion, save_frames
from tqdm import tqdm
import math
from sklearn.cluster import KMeans

def intersection_from_points_and_slopes(b1, m1, b2, m2):
    x1, y1 = 0, b1
    x2, y2 = 0, b2

    if m1 == m2:
        raise ValueError("Lines are parallel and do not intersect.")

    # Solve for x
    x = (m1 * x1 - m2 * x2 + y2 - y1) / (m1 - m2)

    # Solve for y using Line 1
    y = m1 * (x - x1) + y1

    return (x, y)
    

def merge_lines(lines):
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2: continue
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        angle = np.arctan2(y2 - y1, x2 - x1)
        points.append([float(intercept), float(angle)])
        
    x = np.array(points, dtype=np.float32)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(x)
    labels = kmeans.labels_ # Cluster assignments for each point
    centroids = kmeans.cluster_centers_ # Mean of each cluster
    return centroids
        
        
    
def find_court_edges(frames: Generator[np.ndarray, None, None], epsilon = 0.1) -> Generator[np.ndarray, None, None]:
    """Utility generator function to locate court edges on a video feed.
    Applies a visual overlay to the input video feed.
    
        Args:
            frames (typing.Generator[np.ndarray, None, None]): The input video feed as a frame generator. Can be produced with load_frames()
    
        Returns:
            typing.Generator[np.ndarray, None, None]: A new generator of the original video feed with court edge lines as a visual overlay [Width, Height, Color Channel].
    """
    lines = []
    pts = None
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Canny edge detection on the cleaned image
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=100)
    
        out = np.zeros_like(frame)
        
        merged_lines = merge_lines(lines)
        
        pts = []
        for line_1 in merged_lines:
            for line_2 in merged_lines:
                b1, angle_1 = line_1
                b2, angle_2 = line_2
                m1 = math.tan(angle_1)
                m2 = math.tan(angle_2)
                if m1 == m2: continue
                px, py = intersection_from_points_and_slopes(b1, m1, b2, m2)
                if px < 0 or px > frame.shape[0] or py < 0 or py > frame.shape[1]: continue
                pts.append([int(px), int(py)])
                
        print(pts)
        cv2.polylines(out, pts, True, (255, 20, 150))
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            cv2.polylines(out, pts, True, (255, 20, 150))
            #cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (107, 230, 2), thickness=3)
            
        
        
        yield out

def convex_hull(frames: Generator[np.ndarray, None, None], threshold: float = 100) -> Generator[np.ndarray, None, None]:
    for frame in frames:
        # Detect edges using Canny
        canny_output = cv2.Canny(frame, threshold, threshold * 2)
    
        # Find contours
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the convex hull object for each contour
        hulls = [cv2.convexHull(contour) for contour in contours]
        hulls_squeezed = np.concatenate(hulls).squeeze(1)
        group_hull = cv2.convexHull(hulls_squeezed)
        
        # Draw contours + hull results
        hull_frame = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        
        
            
        group_hull = cv2.approxPolyDP(group_hull, 50, True)
        cv2.drawContours(hull_frame, [group_hull], 0, (200, 56, 85), thickness=5)
        
        yield hull_frame
        

        
if __name__ == "__main__":
    threshold = 100
    video_path = os.path.join(os.getcwd(), "videos/louisville_60s_clip.mp4")
    court_segmentation_frames = segment_court_frames(video_path, HF_MODEL)
    occlusion_reduced_frames = reduce_player_occlusion(court_segmentation_frames)
    court_convex_hull_frames = convex_hull(occlusion_reduced_frames)
    #court_edges = find_court_edges(court_convex_hull_frames)
    for frame, overlay in zip(load_frames(video_path), court_convex_hull_frames):
        output = cv2.add(frame, overlay)
        cv2.imshow(__file__, output)