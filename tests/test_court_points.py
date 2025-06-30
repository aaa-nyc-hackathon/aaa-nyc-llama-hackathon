import os
import sys
import cv2
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from experimental.segmentation import sample_frame, court_points
from experimental.config import TMP_DIRNAME_IMAGES

def test_court_points(benchmark):
    assert len(os.listdir(TMP_DIRNAME_IMAGES)) == 0
    frame, filename = sample_frame()
    assert len(os.listdir(TMP_DIRNAME_IMAGES)) != 0
    contour, points = benchmark(court_points, frame)
    assert contour is not None and points is not None
    
if __name__ == "__main__":
    frame, filename = sample_frame()
    floor_contour, floor_points = court_points(
        frame, 
        slic_iters=20, 
        merge_iters=1, 
        t=0.05, 
        sample_size=3
    )
    
    # Draw points and floor to frame
    for point in floor_points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    cv2.drawContours(frame, [floor_contour], -1, (0, 255, 0), 2)
    cv2.imshow(filename, frame)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()