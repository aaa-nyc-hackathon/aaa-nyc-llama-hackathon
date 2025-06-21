from typing_extensions import List
from ultralytics import YOLO
from ultralytics.models.sam.predict import Results
import cv2
import os

import matplotlib.pyplot as plt


def read_batch_frames(cap, batch_size):
    batch = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break  # End of video or read error
        batch.append(frame)
    return batch

def main():
    cap = cv2.VideoCapture(os.path.join(os.getcwd(), "videos/louisville_60s_clip.mp4"))
    model = YOLO('yolov8n.pt')
    prev_head_count = 0
    head_counts = [0]*60
    avg_head_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return  # End of video or read error
            
        
        camera_pov_change = avg_head_count - prev_head_count > 3
        
        if camera_pov_change:
            results = model.predict(frame, show=False)
        else:
            results = model.predict(frame, show=True)
            
        head_count = 0
        if not results[0].boxes:
            continue
            
        head_count = results[0].boxes.xyxy.shape[0]
        avg_head_count = sum(head_counts) / 60
        prev_head_count = head_count
            
        if camera_pov_change:
            continue
        
        head_counts.append(head_count)
        head_counts.pop(0)
        
        
        
        
            
        
        
            
        
    
    

if __name__ == "__main__":
    main()


    
