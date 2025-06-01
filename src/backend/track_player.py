import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

def track_player_points(video_path: str, points: List[Tuple[float, float]], 
                       roi_size: int = 30, 
                       display_only: bool = True) -> None:
    """
    Track a player or object in a video based on initial x,y positions.
    
    Args:
        video_path: Path to the video file
        points: List of (x,y) coordinates to track (normalized 0-1 coordinates)
        roi_size: Size of the square region of interest in pixels (default: 30)
        display_only: If True, only display the video without saving (default: True)
        
    Returns:
        None
    """    # Define parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Convert normalized coordinates to pixel coordinates
    pixel_points = []
    for x, y in points:
        # Convert from normalized (0-1) to pixel coordinates
        if 0 <= x <= 1 and 0 <= y <= 1:
            pixel_x = int(x * frame_width)
            pixel_y = int(y * frame_height)
            pixel_points.append((pixel_x, pixel_y))
        else:
            # Assume these are already pixel coordinates
            pixel_points.append((int(x), int(y)))
    
    # Print debug info
    print(f"Video dimensions: {frame_width}x{frame_height}")
    print(f"Original points: {points}")
    print(f"Pixel points: {pixel_points}")
    
    # Set up window for display
    cv2.namedWindow('Player Tracking', cv2.WINDOW_NORMAL)
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read the first frame from video")
      # Convert to grayscale
    gray_old = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize points to track and their history
    tracking_points = []
    track_history = [[] for _ in range(len(pixel_points))]  # Store history of tracked points
    display_frame = frame.copy()
    
    # Process each initial point
    for point_idx, (x, y) in enumerate(pixel_points):
        # Make sure x and y are within frame boundaries
        x = max(roi_size // 2, min(frame_width - roi_size // 2, x))
        y = max(roi_size // 2, min(frame_height - roi_size // 2, y))
        
        # Extract ROI
        roi_x = max(0, x - roi_size // 2)
        roi_y = max(0, y - roi_size // 2)
        roi_w = min(roi_size, frame_width - roi_x)
        roi_h = min(roi_size, frame_height - roi_y)
        
        roi = gray_old[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Draw initial ROI on display frame
        cv2.rectangle(display_frame, 
                     (roi_x, roi_y), 
                     (roi_x + roi_w, roi_y + roi_h), 
                     (0, 255, 0), 2)
        
        # Find good points to track in ROI
        p = cv2.goodFeaturesToTrack(roi, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        if p is not None and len(p) > 0:
            # Adjust coordinates back to full frame
            p = p + np.array([roi_x, roi_y], dtype=np.float32)
            tracking_points.append(p)
              # Draw the initial point on the display frame
            pt_x, pt_y = p[0].ravel()
            cv2.circle(display_frame, (int(pt_x), int(pt_y)), 5, (0, 0, 255), -1)
    
    # Display the first frame with ROIs and initial points
    cv2.imshow('Player Tracking', display_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        
        # Process each tracking point
        new_tracking_points = []
        
        for idx, p0 in enumerate(tracking_points):
            if p0 is None or len(p0) == 0:
                # Handle loss of tracking point
                new_tracking_points.append(None)
                continue
                
            # Calculate optical flow
            p1, status, _ = cv2.calcOpticalFlowPyrLK(gray_old, gray_new, p0, None, **lk_params)
            
            # If no points were tracked successfully
            if p1 is None or len(p1[status == 1]) == 0:
                # Try to find a new point in the original ROI area
                x, y = pixel_points[idx]
                
                roi_x = max(0, x - roi_size // 2)
                roi_y = max(0, y - roi_size // 2)
                roi_w = min(roi_size, frame_width - roi_x)
                roi_h = min(roi_size, frame_height - roi_y)
                
                roi = gray_new[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                  # Find new points to track
                new_p = cv2.goodFeaturesToTrack(roi, maxCorners=1, qualityLevel=0.3, minDistance=7, blockSize=7)
                
                if new_p is not None and len(new_p) > 0:
                    # Adjust coordinates back to full frame
                    new_p = new_p + np.array([roi_x, roi_y], dtype=np.float32)
                    new_tracking_points.append(new_p)
                    
                    # Get point coordinates
                    pt_x, pt_y = new_p[0].ravel()
                    
                    # Add to history
                    track_history[idx].append((int(pt_x), int(pt_y)))
                    if len(track_history[idx]) > 30:
                        track_history[idx].pop(0)
                    
                    # Draw the new ROI
                    cv2.rectangle(display_frame, 
                               (roi_x, roi_y), 
                               (roi_x + roi_w, roi_y + roi_h), 
                               (0, 255, 255), 2)
                    
                    # Draw the new point
                    cv2.circle(display_frame, (int(pt_x), int(pt_y)), 7, (0, 0, 255), -1)
                else:
                    new_tracking_points.append(None)
            else:
                # Get the good points that were successfully tracked
                good_new = p1[status == 1]
                
                if len(good_new) > 0:
                    new_tracking_points.append(good_new.reshape(-1, 1, 2))
                    
                    # Get the latest tracked position
                    x, y = good_new[0].ravel()
                    
                    # Add point to track history (store last 30 positions)
                    track_history[idx].append((int(x), int(y)))
                    if len(track_history[idx]) > 30:  # Keep only the last 30 points
                        track_history[idx].pop(0)
                    
                    # Draw ROI around the current point
                    roi_x = max(0, int(x) - roi_size // 2)
                    roi_y = max(0, int(y) - roi_size // 2)
                    roi_w = min(roi_size, frame_width - roi_x)
                    roi_h = min(roi_size, frame_height - roi_y)
                    
                    # Draw the trailing path (history of points)
                    for i in range(1, len(track_history[idx])):
                        if track_history[idx][i-1] is not None and track_history[idx][i] is not None:
                            cv2.line(display_frame, track_history[idx][i-1], track_history[idx][i], (0, 255, 0), 2)
                        cv2.rectangle(display_frame, 
                               (roi_x, roi_y), 
                               (roi_x + roi_w, roi_y + roi_h), 
                               (255, 0, 0), 2)
                    
                    # Draw the tracked point (green dot) - make it more prominent
                    cv2.circle(display_frame, (int(x), int(y)), 7, (0, 255, 0), -1)
                else:
                    new_tracking_points.append(None)
          # Update tracking points for next iteration
        tracking_points = new_tracking_points
        
        # Update reference frame
        gray_old = gray_new.copy()
        
        # Display the frame with tracking visualizations
        cv2.imshow('Player Tracking', display_frame)
        
        # Exit if ESC key is pressed or window is closed
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key
            break
        
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    return None


if __name__ == "__main__":
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Path to the video file
    video_file = os.path.join(project_root, "videos", "alabama_clemson_30s_clip.mp4")
    
    # Check if the video file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file not found at {video_file}")
        exit(1)
    else:
        print(f"Video file found: {video_file}")

    # Example JSON input with player coordinates
    json_input = {
        "start": 0, 
        "end": 1, 
        "image_data": [
            {"jerseyNumber": 0, "coordinates": {"x_coordinate": 0.164, "y_coordinate": 0.469}, "team": "Clemson"}, 
            {"jerseyNumber": 2, "coordinates": {"x_coordinate": 0.34, "y_coordinate": 0.503}, "team": "Alabama"}, 
            {"jerseyNumber": 12, "coordinates": {"x_coordinate": 0.475, "y_coordinate": 0.499}, "team": "Alabama"}, 
            {"jerseyNumber": 3, "coordinates": {"x_coordinate": 0.5, "y_coordinate": 0.378}, "team": "Alabama"}, 
            {"jerseyNumber": 5, "coordinates": {"x_coordinate": 0.595, "y_coordinate": 0.446}, "team": "Alabama"}, 
            {"jerseyNumber": 10, "coordinates": {"x_coordinate": 0.794, "y_coordinate": 0.633}, "team": "Clemson"}, 
            {"jerseyNumber": 24, "coordinates": {"x_coordinate": 0.826, "y_coordinate": 0.614}, "team": "Clemson"}
        ]
    }
    
    # Create a list of (x, y) coordinate tuples from the JSON data
    list_of_points = [
        (data["coordinates"]["x_coordinate"], data["coordinates"]["y_coordinate"]) 
        for data in json_input["image_data"]
    ]
    
    try:
        # Call the tracking function to display the video
        print("Starting player tracking visualization. Press ESC to exit.")
        track_player_points(
            video_path=video_file,
            points=list_of_points,
            display_only=True
        )
        print("Tracking visualization complete")
    except Exception as e:
        print(f"Error during tracking: {e}")

