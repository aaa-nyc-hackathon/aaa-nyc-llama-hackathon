import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2
from segmentation import (
    segment_court_frames, 
    simplify_contour, 
    convex_hull, 
    match_points, 
    player_court_mapping,
    reduce_player_occlusion
)
from typing import Generator
import numpy as np
import os
import torch
from config import TMP_DIRNAME_VIDEOS, HF_MODEL
from utils import load_frames
from ultralytics import YOLO


        
if __name__ == "__main__":
    threshold = 100
    video_path_court = os.path.join(os.getcwd(), "videos/louisville_60s_clip.mp4")
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 360))
    frames = load_frames(video_path_court)
    court_segmentation_frames = segment_court_frames(video_path_court, HF_MODEL)
    occlusion_reduced_frames = reduce_player_occlusion(court_segmentation_frames)
    court_convex_hull_frames = convex_hull(occlusion_reduced_frames)
    warped_court_frames = player_court_mapping(court_segmentation_frames, frames)
    model = YOLO('yolov8n.pt')
    # Store previous court points (consumes first frame)
    zipped = zip(court_convex_hull_frames, frames)
    (prev_hull, _), prev_frame = next(zipped)
    prev_court_pts = simplify_contour(prev_hull)
    for (hull, _), frame in zipped:
        court_pts = simplify_contour(hull)
        image_court_points = match_points(prev_court_pts.squeeze(1), court_pts.squeeze(1))
        cv2.circle(frame, image_court_points[0], 5, (0, 0, 255), -1) # top-right
        cv2.circle(frame, image_court_points[1], 5, (255, 0, 255), -1) # bottom-left
        cv2.circle(frame, image_court_points[2], 5, (0, 255, 255), -1) # top-left
        cv2.circle(frame, image_court_points[3], 5, (255, 0, 0), -1) # bottom-right
        
        width = frame.shape[1]
        height = frame.shape[0]
        court_points = np.array([
                 
            [width, height],        # bottom-right
            [0, height],     # bottom-left
            [0, 0], # top-left
            
            [width, 0],             # top-right   
        ], dtype=np.int32)
        
        
        H, _ = cv2.findHomography(image_court_points, court_points)
        
        result = model(frame)[0].boxes.xyxy
        mid_x = (result[:,0] + result[:,2]) // 2
        low_y = result[:,3]
        person_xy = np.expand_dims(torch.stack([mid_x, low_y], dim=-1).numpy().astype(np.int32), axis=1)
        
        # Filter points inside contour
        inside_points = []
        for pt in person_xy:
            x, y = pt.ravel()
            # pointPolygonTest returns:
            # > 0 if inside, = 0 if on edge, < 0 if outside
            if cv2.pointPolygonTest(image_court_points, (int(x), int(y)), measureDist=False) >= 0:
                inside_points.append(pt)
        
        inside_points = np.array(inside_points)
        
        court_points_top_view = cv2.perspectiveTransform(inside_points.astype(np.float32), H).astype(np.int32)
        
        top_frame = np.zeros_like(frame)
        for point in court_points_top_view:
            x, y = point.ravel()
            cv2.circle(top_frame, (int(x), int(y)), 5, (0, 255, 255), -1)
            
        for point in inside_points:
            x, y = point.ravel()
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
            
        output = cv2.drawContours(frame, [image_court_points.reshape([4, 1, 2])], -1, (0, 255, 0), 2)
        out.write(top_frame)
        cv2.imshow("warped frame", output)
        cv2.imshow("top view", top_frame)
        prev_court_pts = court_pts
        
    out.release()
