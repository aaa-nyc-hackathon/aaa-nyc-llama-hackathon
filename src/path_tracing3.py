import cv2
import numpy as np

# Open input video
cap = cv2.VideoCapture("output.mp4")
cap2 = cv2.VideoCapture("videos/alabama_clemson_30s_clip.mp4")

# Output video writer (optional)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Define HSV color ranges for different blobs
# Adjust these based on your blob colors
color_ranges = {
    "purple":  ((120, 50, 50), (150, 255, 255)),
    "green":   ((40, 50, 50),  (80, 255, 255)),
    "red1":    ((0, 50, 50),   (10, 255, 255)),
    "red2":    ((170, 50, 50), (180, 255, 255)),  # red spans across hue wrap
    "blue":    ((100, 50, 50), (130, 255, 255)),
}

# Colors for visualization (BGR)
draw_colors = {
    "purple": (255, 0, 255),
    "green":  (0, 255, 0),
    "red1":   (0, 0, 255),
    "red2":   (0, 0, 255),
    "blue":   (255, 0, 0),
}

while cap.isOpened():
    ret, frame = cap.read()
    ret2, frame_og = cap2.read()
    if not ret or not ret2:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for name, (lower, upper) in color_ranges.items():
        # Create color mask
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        # Optional: clean noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 100:  # skip small blobs
                continue

            # Get centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Draw circle
            cv2.circle(frame, (cx, cy), 10, draw_colors[name], 2)
            cv2.putText(frame, name, (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_colors[name], 1)

    # Write frame to output
    out.write(cv2.add(frame, frame_og))

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
