import sys
from pathlib import Path

from torch.nn.functional import threshold
sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2
from segmentation import segment_court_frames
from typing import Generator
import numpy as np
import random
import os
from config import TMP_DIRNAME_VIDEOS, HF_MODEL
from utils import load_frames, reduce_player_occlusion
from tqdm import tqdm


def court_convex_hull(frames: Generator[np.ndarray, None, None], threshold: float = 100) -> Generator[np.ndarray, None, None]:
    for frame in frames:
        # Detect edges using Canny
        canny_output = cv2.Canny(frame, threshold, threshold * 2)
    
        # Find contours
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the convex hull object for each contour
        largest_hull = None
        max_area = 0
        hulls = []
        
        for i in range(len(contours)):
            hull = cv2.convexHull(contours[i])
            contour_area = cv2.contourArea(hull)
            if contour_area > max_area:
                largest_hull = hull
                max_area = contour_area
                
            hulls.append(hull)
            
        hulls_squeezed = np.concatenate(hulls).squeeze(1)
        group_hull = cv2.convexHull(hulls_squeezed)
        
        # Draw contours + hull results
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        if largest_hull is not None:
            color = (200, 56, 85)
            #cv2.drawContours(drawing, contours, 0, color)
            
            cv2.drawContours(drawing, [group_hull], 0, color, thickness=5)
            
        yield drawing
        
if __name__ == "__main__":
    threshold = 100
    video_path = os.path.join(os.getcwd(), "videos/louisville_60s_clip.mp4")
    out_path = os.path.join(os.getcwd(), f"videos/cache/segmentation/louisville_60s_clip.mp4")
    court_segmentation_frames = segment_court_frames(video_path, HF_MODEL)
    occlusion_reduced_frames = reduce_player_occlusion(court_segmentation_frames)
    court_convex_hull_frames = court_convex_hull(occlusion_reduced_frames, threshold=threshold)
    for frame, overlay in zip(load_frames(video_path), court_convex_hull_frames):
        output = cv2.add(frame, overlay)
        cv2.imshow(__file__, output)