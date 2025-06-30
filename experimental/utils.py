# -*- coding: utf-8 -*-
"""
utils.py

A utility library for video frame preprocessing, court structure detection, and motion tracking
in sports video analytics. This library is designed for analyzing basketball court footage,
especially when the court is partially occluded by players, and aims to enable frame-by-frame
transformations, court geometry estimation, and visualization overlays.

Modules and Functions:
----------------------

- initialize_sam2() -> DeviceLikeType:
    Initializes PyTorch device context for using SAM2, handling CUDA, MPS, and CPU fallbacks.

- load_frames(in_path: str, out_path: str | None = None) -> Generator[np.ndarray, None, None]:
    Generator that reads video frames from an input file. Optionally writes to an output file.

- trace_movement(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    Applies optical flow (Lucas-Kanade) on a video stream and overlays motion trajectories.

- find_court_edges_old(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    Legacy version of court edge detection using Canny and Hough transforms.

- reduce_player_occlusion(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    Applies morphological operations to smooth out and fill gaps caused by occluding players.

- find_court_edges(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    Improved court edge detection that identifies extremal white pixels and draws polygon outlines.

- frame_transformation_template(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    Template function for frame-level transformations to be filled in by the user.

- frame_transformation_pipeline_template(*args, **kwargs):
    Example pipeline that composes the above transformations in sequence and displays the result.

Usage:
------
Run the module directly to execute a sample pipeline on a hardcoded input path:
```bash
python court_utils.py
"""

import cv2
import numpy as np
import os
import torch
from typing import Generator, Callable, TypeVar, Any
from torch._prims_common import DeviceLikeType
import shutil
from pathlib import Path
from typing_extensions import Concatenate, ParamSpec
import sys
import subprocess
from segmentation import reduce_player_occlusion

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import TMP_DIRNAME_IMAGES, MAX_FRAMES, TMP_DIRNAME_VIDEOS

cap = None
P = ParamSpec("P")

def initialize_sam2() -> DeviceLikeType:
    """Utility boilerplate code for loading SAM2 code
    
        Returns:
            torch.DeviceLikeType: The detected available device to run Pytorch with.
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    
    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
        
    return device

def load_frames(in_path: str, out_path: str | None = None) -> Generator[Any, None, None]:
    """Utility generator function to load video (mp4) files
    
        Args:
            in_path (str): The path of the mp4 file to read from.
            out_path (str): The path of the mp4 file to write to. [optional]
    
        Returns:
            typing.Generator[np.ndarray, None, None]: A generator that incrementally yields frames from the mp4 file as a 3-dimensional numpy array [Width, Height, Color Channel].
    """
    # Video reader
    
    global cap
    if cap is None:
        cap = cv2.VideoCapture(in_path)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = None
    if out_path is not None:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
    print(cap.get(cv2.CAP_PROP_POS_FRAMES), cap.isOpened())
    ret, prev_frame = cap.read()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if out is not None:
            yield frame, out.write
        else:
            yield frame
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    

    
    cap.release()
    if out is not None:
        out.release()
        
def generate_frames(video_path: str, output_dir: str, max_frames: int = 500, overwrite=True):
    if len(os.listdir(output_dir)) > 0 and not overwrite:
        raise ValueError(f"Output directory {output_dir} is not empty")
    elif len(os.listdir(output_dir)) > 0 and overwrite:
        print("Warning: writing over cached images")
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        
    # Construct the output file path template
    output_pattern = os.path.join(output_dir, "%05d.jpg")
    
    # Build the ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:v", "2",
        "-start_number", "0",
        "-frames:v", str(max_frames),
        output_pattern
    ]
    
    # Run the command
    subprocess.run(cmd, check=True)
    
def trace_movement(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    """Utility generator function to perform optical flow analysis on video frames.
    Traces moving points' trajectories onto the input video feed
    
        Args:
            frames (typing.Generator[np.ndarray, None, None]): The input video feed as a frame generator. Can be produced with load_frames()
    
        Returns:
            typing.Generator[np.ndarray, None, None]: A new generator of the original video feed with optical flow lines as a visual overlay [Width, Height, Color Channel].
    """
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p0 = None
    old_gray = None
    
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    
    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if p0 is None or old_gray is None:
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            # Create a mask image for drawing purposes
            mask = np.zeros_like(frame)
        
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        
        yield img
            
def frame_transformation_template(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    # Initialized variables before first frame go here ...
    # ...
    for frame in frames:
        # do some transformations to the current frame here
        # ...
        pass
        yield frame
    
def frame_transformation_pipeline_template(*args, **kwargs):
    # By currying different generators into each other, each function is interchangeable with one another
    in_path = os.path.join(os.getcwd(), "videos/alabama_clemson_30s_clip.mp4")
    frames_0 = load_frames(in_path)
    frames_1 = reduce_player_occlusion(frames_0)
    for frame in frames_1:
        cv2.imshow('frame', frame)
    
if __name__ == "__main__":
    frame_transformation_pipeline_template()