# -*- coding: utf-8 -*-
"""
Integration Test: segment_player_frames

This script tests the `segment_player_frames` function from `sam_utils`, which performs instance segmentation 
on players in a basketball video using the SAM2 and YOLO models.

The function generates segmentation masks for detected player bounding boxes and overlays them on the video frames. 
Each resulting frame with an overlay is displayed in a GUI window and optionally written to an output video file 
using the global `utils.out` writer if initialized.

Usage:
    Run the script directly to visualize and optionally save the segmented video.

Requirements:
    - The `HF_MODEL` checkpoint path must be specified in `config.py`.
    - The input video should exist at `videos/louisville_60s_clip.mp4`.
"""

import os
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from experimental import utils, sam_utils
from config import HF_MODEL

if __name__ == "__main__":
    video_path = os.path.join(os.getcwd(), "videos/louisville_60s_clip.mp4")
    out_path = os.path.join(os.getcwd(), "videos/louisville_60s_clip_output.mp4")
    for frame, overlay in sam_utils.segment_player_frames(video_path, None, HF_MODEL):
        output = cv2.add(frame, overlay)
        cv2.imshow('frame', output)
        if not utils.out: continue
        utils.out.write(output)