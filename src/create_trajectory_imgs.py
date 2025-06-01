#!/usr/bin/env python3
import os
import argparse
import cv2 as cv
import numpy as np

def track_and_draw_on_first_frame(
    video_path: str,
    start_time: float,
    end_time: float,
    cx: int,
    cy: int,
    roi_size: int = 50,
    max_distance: float = 100.0,
    output_filename: str = "trajectory.png"
):
    """
    1) Seeks to start_time (sec) in video_path.
    2) On that first frame, finds corners in a square ROI of half‐width=roi_size around (cx, cy).
       If no corners, fall back to (cx, cy).
    3) Reads one more frame (the “second frame”), runs LK flow on all ROI corners, and picks
       the corner with largest displacement to be our tracked_pt. If LK fails on all, fall back to (cx,cy).
    4) From frame 3 onward, tracks that single point frame-by-frame until end_time, collecting its (x,y).
    5) Takes the “very first” frame, draws a single colored line through all collected (x,y), and saves 
       as output_filename.
    """

    # ---- Step A: open video ----
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    # Frames per second (fallback to 30 if grab fails)
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    # Convert seconds → milliseconds
    start_ms = start_time * 1000.0
    end_ms   = end_time   * 1000.0

    # Seek to just after start_ms
    cap.set(cv.CAP_PROP_POS_MSEC, start_ms)

    # ---- Step B: grab the “first” frame ----
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read the first frame at the requested start time.")
        return

    # Convert to grayscale
    first_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    h, w = first_gray.shape

    # ---- Step C: build a mask that is white only in the square ROI around (cx,cy) ----
    x_min = max(0, cx - roi_size)
    y_min = max(0, cy - roi_size)
    x_max = min(w - 1, cx + roi_size)
    y_max = min(h - 1, cy + roi_size)

    mask = np.zeros_like(first_gray, dtype=np.uint8)
    mask[y_min:y_max+1, x_min:x_max+1] = 255

    # ---- Step D: detect up to 100 good corners inside that ROI ----
    feature_params = dict(
        maxCorners  = 100,
        qualityLevel= 0.01,
        minDistance = 7,
        blockSize   = 7
    )
    p0 = cv.goodFeaturesToTrack(
        image = first_gray,
        mask  = mask,
        **feature_params
    )

    # If no corners found, fallback to center (cx,cy) as a single point
    if p0 is None or len(p0) == 0:
        print("⚠️  No corners detected in the ROI. Falling back to (cx, cy).")
        p0 = np.array([[[float(cx), float(cy)]]], dtype=np.float32)

    # ---- Step E: read the “second” frame, so we can choose the one corner with max‐motion ----
    ret, second_frame = cap.read()
    if not ret:
        print("ERROR: Could not read the second frame after the start time.")
        return

    second_gray = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)

    # LK‐flow parameters
    lk_params = dict(
        winSize  = (15, 15),
        maxLevel = 2,
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Compute LK from first_gray → second_gray for all p0
    p1, st, err = cv.calcOpticalFlowPyrLK(
        first_gray,
        second_gray,
        p0,
        None,
        **lk_params
    )

    # Filter out only the “good” corners
    if p1 is None:
        good_new = np.empty((0,2), dtype=np.float32)
        good_old = np.empty((0,2), dtype=np.float32)
    else:
        st_flat  = st.reshape(-1)
        good_new = p1[st_flat == 1].reshape(-1,2)
        good_old = p0[st_flat == 1].reshape(-1,2)

    # If no good corners survived, fallback again to (cx,cy)
    if good_new.shape[0] == 0:
        print("⚠️  LK lost all corners in the second frame. Using (cx,cy) instead.")
        tracked_pt = np.array([cx, cy], dtype=np.float32)
    else:
        # Compute displacement magnitudes
        disps = good_new - good_old  # shape=(N,2)
        mags  = np.linalg.norm(disps, axis=1)
        max_idx = int(np.argmax(mags))
        tracked_pt = good_new[max_idx].astype(np.float32)

    # Convert to the shape (1,1,2) so we can feed it to calcOpticalFlowPyrLK later
    tracked_pt = tracked_pt.reshape(-1,1,2).astype(np.float32)
    prev_pt    = tracked_pt.copy()

    # Initialize a Python list that will hold all integer (x,y) along the trajectory.
    # The first “valid” tracked location is in the second frame.
    # So we record that as our first point.
    first_x, first_y = int(prev_pt[0,0,0]), int(prev_pt[0,0,1])
    trajectory_pts  = [ (first_x, first_y) ]

    prev_gray = second_gray.copy()

    # ---- Step F: process frames 3, 4, 5, … until we reach end_time ----
    while True:
        current_ms = cap.get(cv.CAP_PROP_POS_MSEC)
        if current_ms >= end_ms:
            # We’ve advanced beyond end_time (in ms). Stop.
            break

        ret, frame = cap.read()
        if not ret:
            # No more frames to read
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Track that ONE point (prev_pt) from prev_gray → frame_gray
        p1_new, st_new, err_new = cv.calcOpticalFlowPyrLK(
            prev_gray,
            frame_gray,
            prev_pt,
            None,
            **lk_params
        )

        # If LK fails (p1_new is None or st==0), stop tracking
        if p1_new is None or st_new[0][0] == 0:
            print("⚠️  Lost the tracked point—stopping early.")
            break

        new_pt = p1_new.reshape(-1,2)[0]
        old_pt = prev_pt.reshape(-1,2)[0]

        # If it “jumps” farther than max_distance pixels, we assume bad track and stop
        dist = np.linalg.norm(new_pt - old_pt)
        if dist > max_distance:
            print(f"⚠️  Jump of {dist:.1f}px > max_distance ({max_distance}); stopping.")
            break

        # Record the integer‐rounded coordinate in our list
        trajectory_pts.append( (int(new_pt[0]), int(new_pt[1])) )

        # Prepare for next iteration
        prev_gray = frame_gray.copy()
        prev_pt   = new_pt.reshape(-1,1,2).astype(np.float32)

    cap.release()    # ---- Step G: draw the entire trajectory on a copy of first_frame ----
    canvas = first_frame.copy()

    if len(trajectory_pts) == 1:
        # Only got a single point → draw a single red circle
        x0, y0 = trajectory_pts[0]
        cv.circle(canvas, (x0, y0), 6, (0, 0, 255), -1, lineType=cv.LINE_AA)
    else:
        # Draw line segments (in green) connecting successive points,
        # and draw a small red circle at each point.
        for i in range(1, len(trajectory_pts)):
            x_prev, y_prev = trajectory_pts[i-1]
            x_cur,  y_cur  = trajectory_pts[i]
            cv.line(canvas, (x_prev, y_prev), (x_cur, y_cur), (0, 255, 0), 2, lineType=cv.LINE_AA)
            cv.circle(canvas, (x_cur, y_cur), 4, (0, 0, 255), -1, lineType=cv.LINE_AA)

        # Draw a distinct blue circle at where the track started
        x_start, y_start = trajectory_pts[0]
        cv.circle(canvas, (x_start, y_start), 6, (255, 0, 0), -1, lineType=cv.LINE_AA)
    
    # Draw a bounding box that is 50% larger than the original ROI
    expanded_roi_size = int(roi_size)  # 50% larger than original ROI
    box_x_min = max(0, cx - expanded_roi_size)
    box_y_min = max(0, cy - expanded_roi_size)
    box_x_max = min(w - 1, cx + expanded_roi_size)
    box_y_max = min(h - 1, cy + expanded_roi_size)
    
    # Draw the expanded bounding box in yellow (BGR: 0, 255, 255) with thickness 2
    cv.rectangle(canvas, (box_x_min, box_y_min), (box_x_max, box_y_max), (0, 255, 255), 2, lineType=cv.LINE_AA)

    # ---- Step H: save the single output image ----
    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    cv.imwrite(output_filename, canvas)
    print(f"✅ Done. Trajectory image saved as: {output_filename}")

