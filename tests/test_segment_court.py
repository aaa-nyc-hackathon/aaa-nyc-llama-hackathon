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
from experimental.segmentation import segment_court_frames
from config import HF_MODEL, TMP_DIRNAME_VIDEOS

def segment_court(model: str = HF_MODEL):
    video_path = os.path.join(os.getcwd(), "videos/louisville_60s_clip.mp4")
    for overlay in segment_court_frames(video_path, model):
        cv2.imshow(__file__, overlay)

def test_segment_court():
    assert not os.path.exists(os.path.join(TMP_DIRNAME_VIDEOS, "louisville_60s_clip.mp4"))
    segment_court("facebook/sam2-hiera-tiny")
    assert os.path.exists(os.path.join(TMP_DIRNAME_VIDEOS, "louisville_60s_clip.mp4"))

if __name__ == "__main__":
    segment_court()
        
    