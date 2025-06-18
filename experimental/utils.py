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

def load_frames(in_path: str) -> Generator[Any, None, None]:
    """Utility generator function to load video (mp4) files
    
        Args:
            in_path (str): The path of the mp4 file to read from.
            out_path (str): The path of the mp4 file to write to. [optional]
    
        Returns:
            typing.Generator[np.ndarray, None, None]: A generator that incrementally yields frames from the mp4 file as a 3-dimensional numpy array [Width, Height, Color Channel].
    """
    # Video reader
    global cap
    if not cap:
        cap = cv2.VideoCapture(in_path)
    ret, prev_frame = cap.read()
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        yield frame
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    cap.release()


def save_frames(out_path: str):
    def decorator(func: Callable[Concatenate[str, P], Generator[Any, None, None]]) -> Callable[Concatenate[str, P], Generator[Any, None, None]]:
        def wrapper(in_path: str, *args, **kwargs) -> Generator[None, None, None]:
            global cap
            if not isinstance(cap, cv2.VideoCapture):
                cap = cv2.VideoCapture(in_path)
                
            assert isinstance(cap, cv2.VideoCapture)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            for frame in func(in_path, *args, **kwargs):
                out.write(frame)
                yield frame
                
            out.release()
        return wrapper
    return decorator
        
    
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


def find_court_edges_old(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
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
        new_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=100)
    
        # Draw lines on original frame
        output = frame.copy()
        
        if new_lines is not None:
            for line in new_lines:
                x1, y1, x2, y2 = line[0]
                lines.append([x1, y1])
                lines.append([x2, y2])
                while len(lines) > 140:
                    lines.pop(0)
                    
                    
        if lines is not None:
            rect_a = max(lines, key=lambda x: x[0])
            rect_b = min(lines, key=lambda x: x[0])
            rect_c = max(lines, key=lambda x: x[1])
            rect_d = min(lines, key=lambda x: x[1])
            new_pts = np.array([rect_a, rect_c, rect_b, rect_d])
            
            # Compute pairwise distances: result shape will be (4, 4)
            min_d = 1000
            if pts is not None:
                diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]  # shape (4, 4, 2)
                distances = np.linalg.norm(diff, axis=2)  # Euclidean distances
                min_d = np.min(distances + np.eye(4) * 1000, axis=0) > 200
                
                min_d = min_d.repeat([2]).reshape([4, 2])
                new_pts = np.where(min_d, new_pts, pts)
            
            
            
            pts = new_pts.copy()
            new_pts = new_pts.reshape((-1, 1, 2))
            cv2.polylines(output, [new_pts], True, (255, 0, 255), 2)
            """cv2.circle(output, (rect_a[0], rect_a[1]), 5, (0, 0, 255), -1)
            cv2.circle(output, (rect_c[0], rect_c[1]), 5, (255, 0, 0), -1)
            for line in lines:
                cv2.circle(output, (line[0], line[1]), 5, (0, 0, 100), -1)"""
                
        yield output
                
        

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
        
        
def find_court_edges(frames: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
    """Utility generator function to locate court edges on a video feed.
    Applies a visual overlay to the input video feed.
    
        Args:
            frames (typing.Generator[np.ndarray, None, None]): The input video feed as a frame generator. Can be produced with load_frames()
    
        Returns:
            typing.Generator[np.ndarray, None, None]: A new generator of the original video feed with court edge lines as a visual overlay [Width, Height, Color Channel].
    """
    pts = None
    for frame in frames:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find coordinates of all white pixels (value == 255)
        
        ys, xs = np.where(gray == 255)
        
        if len(ys) > 0:
            # Find index of the pixel with the maximum y-value
            max_y_index = np.argmax(ys)
            max_x_index = np.argmax(xs)
            min_y_index = np.argmin(ys)
            min_x_index = np.argmin(xs)
            bottom_most_point = [xs[max_y_index], ys[max_y_index]]
            right_most_point = [xs[max_x_index], ys[max_x_index]]
            top_most_point = [xs[min_y_index], ys[min_y_index]]
            left_most_point = [xs[min_x_index], ys[min_x_index]]
            new_pts = np.array([bottom_most_point, right_most_point, top_most_point, left_most_point])
            
            # Compute pairwise distances: result shape will be (4, 4)
            min_d = 1000
            if pts is not None:
                diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]  # shape (4, 4, 2)
                distances = np.linalg.norm(diff, axis=2)  # Euclidean distances
                min_d = np.min(distances + np.eye(4) * 1000, axis=0) > 100
                
                min_d = min_d.repeat([2]).reshape([4, 2])
                new_pts = np.where(min_d, new_pts, pts)
            
            
            
            pts = new_pts.copy()
            new_pts = new_pts.reshape((-1, 1, 2))
        
            # Convert to BGR to draw in color
            img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
            # Draw a red circle at the bottom-most white pixel
            cv2.circle(img_color, bottom_most_point, radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(img_color, right_most_point, radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(img_color, top_most_point, radius=5, color=(255, 0, 0), thickness=-1)
            cv2.circle(img_color, left_most_point, radius=5, color=(255, 255, 0), thickness=-1)
            cv2.polylines(img_color, [new_pts], isClosed=True, color=(0, 0, 255), thickness=2)
            
            yield img_color
      
def cache_video(
    func: Callable[Concatenate[str, P], Generator[Any, None, None]]
) -> Callable[Concatenate[str, P], Generator[Any, None, None]]:
    """
    Decorator that loads cached video when possible and provides a frame generator

    This is useful for functions that are called often but are also dependent on other expensive frame operations,
    like SAM segmentation

    Args:
        func (Callable): A generator function that takes an input video path (`in_path`)
        and any additional arguments or keyword arguments, and yields processed frames.

    Returns:
        Callable: A wrapped generator function that extracts frames from a cached video output or another frame generator.
    """
    def wrapper(in_path: str, *args, **kwargs) -> Generator[Any, None, None]:
        out_path = os.path.join(TMP_DIRNAME_VIDEOS, os.path.basename(in_path))
        if os.path.exists(out_path):
            return load_frames(out_path)
        
        if os.path.exists(TMP_DIRNAME_VIDEOS):
            nested_directory_path = Path(TMP_DIRNAME_VIDEOS)
            nested_directory_path.mkdir(parents=True, exist_ok=True)
            
        return save_frames(out_path)(func)(in_path, *args, **kwargs)
    return wrapper

def cache_frames(
    func: Callable[Concatenate[str, P], Generator[Any, None, None]]
) -> Callable[Concatenate[str, P], Generator[Any, None, None]]:
    """
    Decorator that preprocesses video input by saving frames as temporary image files,
    then applies the wrapped generator function to the video input.

    This is useful for functions that operate on video frames and might benefit
    from having those frames saved for debugging, visualization, or intermediate
    processing.

    Args:
        func (Callable): A generator function that takes an input video path (`in_path`),
            an optional output path (`out_path`), and any additional arguments or keyword
            arguments, and yields processed frames.

    Returns:
        Callable: A wrapped generator function that extracts and saves frames
            to a temporary directory (if they don’t already exist), calls the original
            generator function

    Side Effects:
        - Creates a nested temporary directory (if it doesn't exist) to store
            extracted video frames as JPEG images.
    """
    def wrapper(in_path: str, *args, **kwargs) -> Generator[Any, None, None]:
        if not os.path.exists(TMP_DIRNAME_IMAGES):
            nested_directory_path = Path(TMP_DIRNAME_IMAGES)
            nested_directory_path.mkdir(parents=True, exist_ok=True)
            for frame_idx, frame in enumerate(load_frames(in_path)):
                if frame_idx > MAX_FRAMES: continue
                cv2.imwrite(f"{TMP_DIRNAME_IMAGES}/{frame_idx:05d}.jpg", frame)
            
        return func(in_path, *args, **kwargs)
    return wrapper
    
def cleanup_frames(
    func: Callable[Concatenate[str, P], Generator[Any, None, None]]
) -> Callable[Concatenate[str, P], Generator[Any, None, None]]:
    """
    Decorator that preprocesses video input by saving frames as temporary image files,
    then applies the wrapped generator function to the video input, and finally
    cleans up the temporary files.

    This is useful for functions that operate on video frames and might benefit
    from having those frames saved for debugging, visualization, or intermediate
    processing.

    Args:
        func (Callable): A generator function that takes an input video path (`in_path`),
            an optional output path (`out_path`), and any additional arguments or keyword
            arguments, and yields processed frames.

    Returns:
        Callable: A wrapped generator function that deletes the temporary directory after
            the generator is exhausted. Must be used with cache_frames() decorator.

    Side Effects:
        - Deletes the temporary directory upon completion.
    """
    def wrapper(in_path: str, *args, **kwargs) -> Generator[Any, None, None]:
        for x in func(in_path, *args, **kwargs):
            yield x
            
        if os.path.exists(TMP_DIRNAME_IMAGES):
            shutil.rmtree(TMP_DIRNAME_IMAGES)
        
    return wrapper
            
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
    frames_2 = find_court_edges(frames_1)
    for frame in frames_2:
        cv2.imshow('frame', frame)
        
    
if __name__ == "__main__":
    frame_transformation_pipeline_template()