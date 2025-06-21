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
from experimental.segmentation import segment_player_frames
from config import HF_MODEL, TMP_DIRNAME_VIDEOS

def segment_players():
    video_path = os.path.join(os.getcwd(), "videos/louisville_60s_clip.mp4")
    for overlay in segment_player_frames(video_path, HF_MODEL):
        cv2.imshow(__file__, overlay)

def test_segment_players():
    assert not os.path.exists(os.path.join(TMP_DIRNAME_VIDEOS, "louisville_60s_clip.mp4"))
    segment_players()
    assert not os.path.exists(os.path.join(TMP_DIRNAME_VIDEOS, "louisville_60s_clip.mp4"))
    
if __name__ == "__main__":
    segment_players()