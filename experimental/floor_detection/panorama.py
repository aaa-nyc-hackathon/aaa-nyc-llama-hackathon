import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_court_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 100, 100])  # Tune as needed
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask

orb = cv2.ORB_create()

def estimate_transform(prev_frame, curr_frame):
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(curr_frame, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    return None

def estimate_optical_transform(prev_gray, curr_gray):
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, maxCorners=200, qualityLevel=0.01, minDistance=30)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)

    if p1 is not None and st.sum() > 20:
        p0, p1 = p0[st == 1], p1[st == 1]
        H, _ = cv2.estimateAffinePartial2D(p0, p1)
        return H
    return None

#TODO
def reduce_player_occlusion(mask_with_holes):
    inpainted = cv2.inpaint(panorama, mask_with_holes, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def build_panorama(video_stream, show=False, verbose=False):
    frame_count = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    ret, frame = video_stream.read()

    panorama = np.zeros((1000, 2000, 3), dtype=np.uint8)  # Adjust size to fit full court
    H_global = np.eye(3)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in tqdm(range(int(frame_count) - 1)):
        ret, frame = video_stream.read()

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret is None:
            break

        mask = extract_court_mask(frame)


        # Warp the current court region to panorama coordinates
        warped = cv2.warpPerspective(frame, H_global, (panorama.shape[1], panorama.shape[0]))
        warped_mask = cv2.warpPerspective(mask, H_global, (panorama.shape[1], panorama.shape[0]))

        # Blend new warped court into panorama
        panorama[warped_mask > 0] = warped[warped_mask > 0]

        # Read next frame and estimate transform
        H_rel = estimate_transform(prev_gray, curr_gray)
        if H_rel is not None:
            H_global = H_global @ H_rel

        if show:
            cv2.imshow("frame", panorama)

        if verbose:
            print(panorama)


        prev_gray = curr_gray.copy()

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    return panorama


if __name__ == "__main__":
    video_stream = cv2.VideoCapture(os.path.join(os.getcwd(), "experimental/floor_detection/outputs/louisville_60s_clip_segmentation.mp4"))
    panorama = build_panorama(video_stream, show=True, verbose=True)
    video_stream.release()
