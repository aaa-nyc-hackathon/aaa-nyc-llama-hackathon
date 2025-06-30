import numpy as np
import cv2 as cv

def softmax(x, axis=-1):
    """Apply softmax to a numpy array along a specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def apply_softmax_to_mat(matlike, axis=-1):
    """
    Apply softmax to an OpenCV Mat-like (NumPy array).
    
    Parameters:
        matlike: OpenCV Mat (i.e. np.ndarray)
        axis: axis along which to apply softmax (e.g., -1 for channels)
    
    Returns:
        NumPy array suitable for OpenCV (Mat-like)
    """
    # Step 1: Ensure it's a float NumPy array
    arr = np.asarray(matlike).astype(np.float32)

    # Step 2: Apply softmax
    softmaxed = softmax(arr, axis=axis)

    # Step 3: Return as "Mat-like" (still a NumPy array, ready for OpenCV)
    return softmaxed.astype(np.float32)

cap = cv.VideoCapture(cv.samples.findFile("videos/alabama_clemson_30s_clip.mp4"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
flow = None

# Get video properties
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv.CAP_PROP_FPS)

# Define codec and output video file
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or 'avc1'
out = cv.VideoWriter("output.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(next, threshold1=100, threshold2=200, apertureSize=7)
    
    board = cv.Canny(next, threshold1=70, threshold2=200, apertureSize=7)
    
    # Define kernel (structuring element)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    
    # Apply morphological closing (dilate then erode)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # Find contours from the edge image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create a blank canvas to draw filled shapes
    filled = np.zeros_like(frame1)  # Optional: color background
    
    # Draw and fill each contour
    shapes = cv.drawContours(filled, contours, -1, (0, 255, 0), thickness=cv.FILLED)
    
    gray = cv.cvtColor(shapes, cv.COLOR_BGR2GRAY)
    
    # Threshold to get binary mask
    _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    
    intersection = cv.bitwise_xor(binary, edges)
    #cv.imshow("test", binary)
    
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 10, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    gray_hsl = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    
    _, binary_hsl = cv.threshold(gray_hsl, 1, 255, cv.THRESH_BINARY)
    bgr_2 = cv.add(bgr, frame2)
    cv.imshow('frame2', bgr)
    
    #cv.imshow("edges", filled)
    
    # Show edges (black background with white edges)
    #cv.imshow('Edges.png', edges)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame2)
        cv.imwrite('opticalhsv.png', bgr_2)
    prvs = next
    out.write(bgr)
    
cap.release()
out.release()
cv.destroyAllWindows()