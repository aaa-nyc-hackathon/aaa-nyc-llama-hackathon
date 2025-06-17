# -*- coding: utf-8 -*-
"""
Integration Test: segment_court_frames

This script tests the `segment_court_frames` function from `sam_utils`, which performs segmentation 
on the basketball court using the SAM2 video segmentation model.

The function extracts the court shape from the initial video frame and propagates that mask through the 
entire video, generating overlays to visualize court segmentation across frames. The resulting video frames 
with overlays are displayed in a GUI window and optionally saved to an output file if `utils.out` is initialized.

Usage:
    Run the script directly to visualize and optionally save the court-segmented video.

Requirements:
    - The `HF_MODEL` checkpoint path must be defined in `config.py`.
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
    for frame, overlay in sam_utils.segment_court_frames(video_path, out_path, HF_MODEL):
        output = cv2.add(frame, overlay)
        cv2.imshow('frame', overlay)
        if not utils.out: continue
        utils.out.write(overlay)