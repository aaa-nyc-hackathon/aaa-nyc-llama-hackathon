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
from sam2.build_sam import build_sam2_hf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_video_predictor import SAM2VideoPredictor
from typing import Generator
import random
from ultralytics import YOLO
from config import TMP_DIRNAME_IMAGES
from experimental.utils import cache_frames, cache_video, cleanup_frames
from experimental import utils

device = utils.initialize_sam2()

def segment_initial_court_frame(sam2_checkpoint: str, img_path) -> list[int]:
    """
    Performs initial segmentation of the court from a static frame using SAM2.

    Args:
        sam2_checkpoint (str): Path or HuggingFace ID of the SAM2 model checkpoint.
        img_path (str): File path to the image used for initial court segmentation.

    Returns:
        list[int]: A list of 2D point coordinates that define the segmented court region.

    Notes:
        - Only the second largest mask is returned based on area.
        - Assumes the first frame contains a clear view of the court.
    """
    sam2 = build_sam2_hf(sam2_checkpoint, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, 
        stability_score_thresh=0.95,
        stability_score_offset=0.1,
        points_per_side=8
    )
    
    # NOTE: We use a single call of SAM full mask segmentation to locate the court
    # We only extract the initial position of the court from the first frame of the video
    # This will become problematic in production when providing videos that may not start with an ideal frame of the court    
    initial_frame = cv2.imread(img_path)                       
    masks = mask_generator.generate(initial_frame)
    masks.sort(key=lambda x: x['area'])
    points = masks[-2]['point_coords']
    
    return points



def segment_frames(predictor_iter, video_path: str) -> Generator[np.ndarray, None, None]:
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
    for (out_frame_idx, out_obj_ids, out_mask_logits), frame in zip(predictor_iter, utils.load_frames(video_path)):
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
                
        yield overlay

@cache_video
@cleanup_frames
@cache_frames
def segment_court_frames(video_path: str, sam2_checkpoint: str) -> Generator[np.ndarray, None, None]:
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
    inference_state = predictor.init_state(video_path=TMP_DIRNAME_IMAGES)
    predictor.reset_state(inference_state)
    # Use SAM2 mask segmentation to determine the initial location of the court
    points = segment_initial_court_frame(sam2_checkpoint, f"{TMP_DIRNAME_IMAGES}/00001.jpg")
    
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=np.array(points, dtype=np.float32),
        labels=np.array([1 for _ in range(len(points))])
    )
    
    
    return segment_frames(predictor.propagate_in_video(inference_state), video_path)
   
@cache_video     
@cleanup_frames    
@cache_frames
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
        