# -*- coding: utf-8 -*-
"""
Video segmentation utility using SAM2 and YOLO.

This script includes utilities for preprocessing, segmenting courts and players in basketball
videos using Meta’s SAM2 model and YOLO object detection. Frames are temporarily stored to
disk for efficient access and post-processing.

Core functionality:
- Extract and store frames from input video temporarily.
- Use SAM2 to segment the court and track it over time.
- Use YOLO to detect players and propagate their masks.
- Overlay segmentations and optionally save the output.

Dependencies:
- SAM2: Segment Anything Model v2
- YOLOv8: Ultralytics object detection
- OpenCV, NumPy, Torch

Run the script directly to visualize player segmentation from a video.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_video_predictor import SAM2VideoPredictor
from typing import Generator, Callable
import random
from ultralytics import YOLO
from config import TMP_DIRNAME_IMAGES, HF_MODEL, TMP_DIRNAME_VIDEOS
from experimental.utils import generate_frames, load_frames
from experimental import utils
import os
import torch
from collections import defaultdict
import networkx as nx
from scipy.optimize import linear_sum_assignment

device = utils.initialize_sam2()

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]
            u = self.parent[u]
        return u

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv

def sample_floor_points(contour, shape, sample_size):
    # Create a binary mask for the contour
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)  # Filled contour
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Get all pixels inside the contour
    ys, xs = np.where(mask == 255)
    coords = np.stack((xs, ys), axis=-1)
    
    if len(coords) < sample_size:
        raise ValueError(f"Contour only contains {len(coords)} pixels, can't sample {sample_size} points.")
    
    indices = np.random.choice(len(coords), size=sample_size, replace=False)
    return coords[indices]

def floor_segmentation(input: np.ndarray, slic_iters: int, merge_iters: int, t: int | float) -> cv2.typing.MatLike:
    frame = input.copy()
    # 1) Extraction of SLIC superpixels
    slic = cv2.ximgproc.createSuperpixelSLIC(frame) 
    slic.iterate(slic_iters)
    # Get the labels and contour mask
    labels = slic.getLabels() # shape: (H, W), int32
    num_labels = slic.getNumberOfSuperpixels()
    largest_contour = None
    # Iteratively merge small segments and recompute largest segment
    for i in range(merge_iters):
        # 2) Transforming the RGB input image (a) into HSV color space and obtain an edge mapping using saturation channel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(hsv[:,:,1], 100, 200) 
        
        # 3) Constructing a region of adjacency graph (RAG) (f) from the combination of the superpixels image and the edge map.
        neighbors = build_region_adjacency_list(labels) # pixel neighbor adjacency list
        rag = build_region_adjacency_graph(edges, labels, num_labels, neighbors)
        
        # 4) Hierarchical merging of the RAG and final image clusterization (c)
        weights = [data['weight'] for u, v, data in rag.edges(data=True) if 'weight' in data]
        max_weight = max(weights)
        min_weight = min(weights)
        # avg_weight = sum(weights) / len(weights)
        lerp_threshold = t * (max_weight - min_weight) + min_weight
        labels, num_labels = merge_regions(labels, num_labels, rag, lerp_threshold)
            
        # 5) Find largest contour
        contours_list = gather_contours(labels)
        largest_contour = sorted(contours_list, key=cv2.contourArea)[-1] # smallest to largest
        
        # Update output frame
        merged_vis = np.zeros_like(frame)
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        for i in range(num_labels):
            merged_vis[labels == i] = colors[i]
        frame = merged_vis
        
    assert largest_contour is not None # type safety
    return largest_contour

def gather_contours(labels):
    contours_list = []
    for label in np.unique(labels):
            # Create binary mask for the current label
        mask = (labels == label).astype(np.uint8) * 255
        
            # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        for contour in contours:
            contours_list.append(contour)
    return contours_list

def merge_regions(labels: cv2.typing.MatLike, num_labels, rag, threshold):
    uf = UnionFind(num_labels)
    sorted_edges = sorted(rag.edges(data=True), key=lambda x: x[2]['weight']) # sort edges by weight
    for u, v, data in sorted_edges:
        if data['weight'] < threshold:
            uf.union(u, v)
    
    # Relabel the superpixels in the label map
    old_to_new = {}
    new_label = 0
    labels_copy = labels.copy()
    
    for old_label in range(rag.number_of_nodes()):
        root = uf.find(old_label)
        if root not in old_to_new:
            old_to_new[root] = new_label
            new_label += 1
        labels_copy[labels == old_label] = old_to_new[root]
    return labels_copy, new_label

def build_region_adjacency_list(labels):
    h, w = labels.shape
    neighbors = defaultdict(set)
    for current_y in range(h - 1):
        for current_x in range(w - 1):
            current_label = labels[current_y, current_x]
            for neighbor_y, neighbor_x in [(0, 1), (1, 0)]:
                neighbor_label = labels[current_y + neighbor_y, current_x + neighbor_x]
                if current_label != neighbor_label:
                    neighbors[current_label].add(neighbor_label)
                    neighbors[neighbor_label].add(current_label)
                    
    return neighbors

def build_region_adjacency_graph(edges, labels, num_labels, neighbors):
    rag = nx.Graph()
    for label in range(num_labels):
        rag.add_node(label)
    
    for a, adj in neighbors.items():
        for b in adj:
            if rag.has_edge(a, b) or a == b:
                continue
                
            # Make a binary mask for shared boundary pixels
            mask = np.zeros_like(labels, dtype=np.uint8)
            mask[(labels == a) | (labels == b)] = 255
    
            # Count how many edge pixels (from Canny) lie on the boundary
            edge_strength = np.count_nonzero(edges & (mask == 255))
    
            # Add weighted edge to the graph
            rag.add_edge(a, b, weight=edge_strength)
            
    return rag

def sample_frame():
    file_count = len(os.listdir(TMP_DIRNAME_IMAGES))
    if file_count == 0:
        generate_frames(
            video_path=os.path.join(os.getcwd(), "videos/alabama_clemson.mp4"),
            output_dir=TMP_DIRNAME_IMAGES,
            max_frames=2000
        )
        file_count = len(os.listdir(TMP_DIRNAME_IMAGES))
        
    idx = random.randrange(0, file_count)
    filename = os.path.join(TMP_DIRNAME_IMAGES, f"{idx:05d}.jpg")
    print(f"Sampling image: {filename}")
    out = cv2.imread(filename)
    return out, filename
    
def court_points(frame, slic_iters=20, merge_iters=1, t=0.05, sample_size=3):
    floor_contour = floor_segmentation(frame, slic_iters, merge_iters, t)
    floor_points = sample_floor_points(floor_contour, frame.shape, sample_size)
    return floor_contour, floor_points

def convex_hull(frames: Generator[np.ndarray, None, None], threshold: float = 100) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    for frame in frames:
        # Detect edges using Canny
        canny_output = cv2.Canny(frame, threshold, threshold * 2)
    
        # Find contours
        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the convex hull object for each contour
        hulls = [cv2.convexHull(contour) for contour in contours]
        hulls_squeezed = np.concatenate(hulls).squeeze(1)
        group_hull = cv2.convexHull(hulls_squeezed)
        
        yield group_hull, frame
      
def player_boxes(frames: Generator[np.ndarray, None, None]) -> Generator[torch.Tensor, None, None]:
    model = YOLO("yolov8n.pt")
    for frame in frames:
        result = model(frame)
        yield result[0].boxes.xyxy
        
def player_court_mapping(court_segmentation_frames: Generator[np.ndarray, None, None], frames: Generator[np.ndarray, None, None]):
    #boxes = player_boxes(frames)
    occlusion_reduced_frames = reduce_player_occlusion(court_segmentation_frames)
    court_convex_hull_frames = convex_hull(occlusion_reduced_frames)
    
    
    for (hull, _), frame in zip(court_convex_hull_frames, frames):
        result_frame = np.zeros_like(frame)
        '''cv2.drawContours(result_frame, [hull], -1, (0, 255, 0), 2)
        for xyxy in player_frame.unbind():
            mid_x = (xyxy[0].item() + xyxy[2].item()) // 2
            low_y = int(xyxy[1].item())
            cv2.circle(result_frame, (mid_x, low_y), radius=5, color=(0, 0, 255))'''
        hull_squeezed = hull.squeeze(1)
        image_court_points = reduce_court_points(hull_squeezed, frame)
        
        width = frame.shape[1]
        height = frame.shape[0]
        court_points = np.array([
            [0, 0],             # top-left
            [width, 0],         # top-right
            [width, height],    # bottom-right
            [0, height],        # bottom-left
        ], dtype=np.int32)
        
        
        H, _ = cv2.findHomography(image_court_points, court_points)
        warped = cv2.warpPerspective(frame, H, (width, height))
        
        yield hull
      

def reduce_court_points(hull: cv2.typing.MatLike, frame: cv2.typing.MatLike):
    zeros = np.zeros_like(frame)
    zeros = cv2.drawContours(zeros, [hull.squeeze(1)], -1, (0, 255, 0), 2)
    zeros = cv2.cvtColor(zeros, cv2.COLOR_BGR2GRAY)
    court_pts = cv2.goodFeaturesToTrack(zeros, 4, 1e-3 , 50, useHarrisDetector=True, blockSize=20)  # get 4 corners
    assert len(court_pts) == 4 # probably need to account for corners not being found
    return court_pts.squeeze(1).astype(np.int32)

# Sort centers into consistent order (e.g., clockwise from top-left)
def sort_corners(pts):
    """
    Sorts 4 (x, y) points into consistent order:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = np.array(pts, dtype=np.float32)

    # Sum and difference of points
    s = pts.sum(axis=1)            # x + y
    diff = np.diff(pts, axis=1)    # x - y

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    return ordered

# prev_points and curr_points should each be (4, 2) arrays
def match_points(prev_points, curr_points):
    prev = np.array(prev_points)
    curr = np.array(curr_points)

    # Compute pairwise Euclidean distances
    cost_matrix = np.linalg.norm(prev[:, None, :] - curr[None, :, :], axis=2)

    # Hungarian algorithm: find optimal point assignment
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    # Now col_idx[i] tells us which point in curr matches prev[i]
    matched_curr_points = curr[col_idx].astype(np.int32)

    return matched_curr_points  # ordered to match prev_points
        
def simplify_contour(contour, n_corners=4):
    '''
    Binary searches best `epsilon` value to force contour 
        approximation contain exactly `n_corners` points.
        
    :param contour: OpenCV2 contour.
    :param n_corners: Number of corners (points) the contour must contain.
    
    :returns: Simplified contour in successful case. Otherwise returns initial contour.
    '''
    n_iter, max_iter = 0, 100
    lb, ub = 0., 1.
    
    while True:
        n_iter += 1
        if n_iter > max_iter:
            return contour
        
        k = (lb + ub)/2.
        eps = k*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        
        if len(approx) > n_corners:
            lb = (lb + ub)/2.
        elif len(approx) < n_corners:
            ub = (lb + ub)/2.
        else:
            return approx

def reduce_player_occlusion(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    """Utility generator function to remove player and crowd occlusion of the court.
    Applies a visual overlay to the input video feed.
    
        Args:
            frames (typing.Generator[np.ndarray, None, None]): The input video feed as a frame generator. Can be produced with load_frames()
    
        Returns:
            typing.Generator[np.ndarray, None, None]: A new generator of the original video feed with with court edges smoothed and interior holes of the court reduced [Width, Height, Color Channel].
    """
    # Structuring element for morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 50))  # Size adjusts gap-filling power
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        # Threshold to isolate main surfaces (e.g., bright court)
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
        # Fill small holes: Morphological closing
        filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
        out = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)
            
        yield out

def segment_frames(predictor_iter, video_path: str, out_path: str | None = None) -> Generator:
    """
    Iteratively generates video frames with segmentation overlays from a SAM2 predictor.

    Args:
        predictor_iter (Iterable): An iterator from the SAM2 predictor's `propagate_in_video`.
        video_path (str): Path to the original input video.
        out_path (str | None, optional): Optional output path for writing segmented frames.

    Yields:
        Generator[tuple[np.ndarray, np.ndarray], None, None]: A tuple of (original frame, overlay mask).

    Notes:
        - Assigns random colors to each tracked object.
        - Applies binary masks per object ID and blends them on top of the frame.
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    color_lookup = {}
    for (out_frame_idx, out_obj_ids, out_mask_logits), x in zip(predictor_iter, utils.load_frames(video_path, out_path=out_path)):
        write = None
        if out_path is None:
            frame = x
        else:
            frame, write = x[0], x[1]
        
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
        overlay = np.zeros_like(frame, dtype=np.uint8)
        for id in out_obj_ids:
            mask_bool = video_segments[out_frame_idx][id].squeeze(0)
            if id not in color_lookup:
                color_lookup[id] = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
            
            mask_color = color_lookup[id]
            colored_mask = np.zeros_like(frame, dtype=np.uint8)
            colored_mask[:] = mask_color
            alpha = 0.5
            try:
                overlay[mask_bool] = colored_mask[mask_bool]
            except:
                continue
        
        if write is not None:
            write(overlay)
        
        yield overlay


def segment_court_frames(video_path: str, sam2_checkpoint: str, out_path: str | None = None) -> Generator[np.ndarray, None, None]:
    """
    Segments and tracks the basketball court across a video using SAM2.

    This function uses an initial frame to detect the court region and propagates
    the mask across the video.

    Args:
        video_path (str): Path to the input video.
        out_path (str | None): Optional path to save the output video.
        sam2_checkpoint (str): Path or ID of the SAM2 model checkpoint.

    Returns:
        Generator[tuple[np.ndarray, np.ndarray], None, None]: Yields original frames and overlay masks.
    """
    
    predictor = SAM2VideoPredictor.from_pretrained(sam2_checkpoint, device=device)
    # Initialize inference state of SAM2 video model
    generate_frames(video_path, TMP_DIRNAME_IMAGES)
    inference_state = predictor.init_state(video_path=TMP_DIRNAME_IMAGES)
    predictor.reset_state(inference_state)
    # Use SAM2 mask segmentation to determine the initial location of the court
    initial_frame = cv2.imread(f"{TMP_DIRNAME_IMAGES}/00001.jpg")
    sample_size = 3
    _, points = court_points(initial_frame, sample_size=sample_size)
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=points,
        labels=np.array([1]*sample_size)
    )
    
    return segment_frames(predictor.propagate_in_video(inference_state), video_path, out_path=out_path)
  

def segment_player_frames(video_path: str, sam2_checkpoint: str) -> Generator[np.ndarray, None, None]:
    """
    Detects and tracks players across frames using YOLO and SAM2.

    Player bounding boxes are detected using YOLOv8 and then refined and
    propagated using SAM2 segmentation masks across the video.

    Args:
        video_path (str): Path to the input video.
        out_path (str | None): Optional output path to save segmented video.
        sam2_checkpoint (str): Path or ID of the SAM2 model checkpoint.

    Returns:
        Generator[tuple[np.ndarray, np.ndarray], None, None]: Yields original frames and player overlays.
    """
    predictor = SAM2VideoPredictor.from_pretrained(sam2_checkpoint, device=device)
    # Initialize inference state of SAM2 video model
    generate_frames(video_path, TMP_DIRNAME_IMAGES)
    inference_state = predictor.init_state(video_path=TMP_DIRNAME_IMAGES)
    predictor.reset_state(inference_state)
    
    model = YOLO('yolov8n.pt')
    results = model(f"{TMP_DIRNAME_IMAGES}/00001.jpg")
    
    for obj_id, box in enumerate(results[0].boxes.xyxy.unbind()):
        np_box = box.numpy()
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=np_box,
        )
    
    return segment_frames(predictor.propagate_in_video(inference_state), video_path)
    
def cache(watch_path: str, func: Callable, *args, **kwargs):
    if os.path.exists(watch_path):
        return load_frames(watch_path)
    
    return func(*args, **kwargs, out_path=watch_path)

if __name__ == "__main__":
    video_path = os.path.join(os.getcwd(), "videos/soccer_highlight_1_clip.mp4")
    out_path = f"{TMP_DIRNAME_VIDEOS}/soccer_highlight_1_clip.mp4"
    for seg in cache(out_path, segment_court_frames, video_path, HF_MODEL):
        cv2.imshow("court", seg)
