# video_action_classifier.py
import cv2
import base64
import os
import json
import time
import numpy as np
import io
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image
from llama_api_client import LlamaAPIClient
from dotenv import load_dotenv
from scipy.signal import find_peaks

load_dotenv()

class VideoActionClassifier:
    """
    A complete solution for sports video analysis using Llama 4's multimodal capabilities
    with advanced frame sampling techniques.
    """
    
    def __init__(self, model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", max_frames: int = 8):
        """
        Initialize the classifier with Llama API client.
        
        Args:
            model (str): Llama model to use for analysis
            max_frames (int): Maximum number of frames to analyze
        """
        self.client = LlamaAPIClient()
        self.model = model
        self.max_frames = max_frames

    def adaptive_frame_sampling(self, video_path: str, target_frames: int = 5) -> List[np.ndarray]:
        """
        Intelligent frame sampling combining motion detection and keyframe analysis.
        
        Args:
            video_path (str): Path to video file
            target_frames (int): Number of frames to extract
            
        Returns:
            List[np.ndarray]: Selected frames in RGB format
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        motion_scores = []
        prev_gray = None
        frame_count = 0
        
        print(f"Starting first pass: Motion score calculation from {video_path}")
        # First pass: Calculate motion scores
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processing frame {frame_count} for motion detection...")
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
                motion_scores.append(motion)
            else:
                motion_scores.append(0)
            
            prev_gray = gray
            if len(motion_scores) >= 1000:  # Limit processing for long videos
                print(f"Reached 1000 frames limit. Stopping first pass.")
                break
                
        cap.release()
        print(f"First pass complete. Processed {frame_count} frames.")
        
        # Identify high-motion segments
        peaks, _ = find_peaks(motion_scores, distance=10, prominence=np.std(motion_scores))
        print(f"Found {len(peaks)} potential high-motion segments.")
        
        # Second pass: Capture keyframes from high-motion segments
        print(f"Starting second pass: Capturing keyframes from high-motion segments")
        cap = cv2.VideoCapture(video_path)
        selected_frames = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 100 == 0:
                print(f"Second pass: Processing frame {frame_idx}...")
                
            if frame_idx in peaks[:target_frames*2]:
                selected_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print(f"Selected frame {frame_idx} as a keyframe. Total selected: {len(selected_frames)}")
                
            frame_idx += 1
            if len(selected_frames) >= target_frames*2:
                print(f"Reached target of {target_frames*2} selected frames. Stopping second pass.")
                break
                
        cap.release()
        print(f"Second pass complete. Processed {frame_idx} frames.")
        
        # Select most diverse frames using histogram analysis
        print("Analyzing frame diversity using histograms...")
        histograms = [cv2.calcHist([f], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]).flatten() 
                     for f in selected_frames]
        similarities = np.array([[cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA) 
                                for h2 in histograms] for h1 in histograms])
        np.fill_diagonal(similarities, 0)
        diversity_scores = np.sum(similarities, axis=1)
        
        final_frames = [selected_frames[i] for i in diversity_scores.argsort()[-target_frames:]]
        print(f"Final selection: {len(final_frames)} most diverse frames extracted.")
        
        return final_frames

    def frame_to_b64(self, frame: np.ndarray) -> str:
        """Convert frame to optimized base64 string"""
        img = Image.fromarray(frame)
        img.thumbnail((1024, 1024))  # Maintain aspect ratio
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def create_analysis_prompt(self, actions: List[str], sport: str = "soccer") -> str:
        """Generate dynamic prompt based on target actions"""
        return f"""Analyze these video frames from a {sport} match. Identify and describe:
- Specific actions: {', '.join(actions)}
- Player positions and movements
- Ball trajectory and key events
- Temporal sequence of actions

Format response as JSON with:
{{
  "actions_detected": [
    {{
      "action": str,
      "confidence": 0-100,
      "start_frame": int,
      "end_frame": int,
      "description": str  
    }}
  ],  "match_summary": str,
  "key_moments": [str]
}}"""

    def analyze_video(self, video_path: str, actions: List[str], sport: str) -> Dict:
        """
        Full video analysis pipeline.
        
        Args:
            video_path (str): Path to video file
            actions (List[str]): Actions to detect
            sport (str): Sport type for context
            
        Returns:
            Dict: Analysis results from Llama 4
        """
        print(f"\n======== Starting video analysis for {video_path} ========")
        print(f"Target actions to detect: {', '.join(actions)}")
        print(f"Sport context: {sport}")
        
        # Frame extraction
        print(f"Starting frame extraction with target of {self.max_frames} frames...")
        start_time = time.time()
        frames = self.adaptive_frame_sampling(video_path, self.max_frames)
        extraction_time = time.time() - start_time
        print(f"Frame extraction complete. Extracted {len(frames)} frames in {extraction_time:.2f} seconds.")
        
        # Prepare multimodal payload
        print("Preparing multimodal payload...")
        print(f"Creating analysis prompt for {sport} with {len(actions)} target actions")
        content = [{"type": "text", "text": self.create_analysis_prompt(actions, sport)}]
        
        print("Converting frames to base64 format...")
        start_time = time.time()
        for i, frame in enumerate(frames):
            b64_image = self.frame_to_b64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
            })
            print(f"  Processed frame {i+1}/{len(frames)}")
        
        conversion_time = time.time() - start_time
        print(f"Frame conversion complete in {conversion_time:.2f} seconds.")
        
        # API request using Llama API client
        print(f"Sending request to Llama API using model: {self.model}")
        print("This may take some time depending on model load and queue...")
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": content}],
                model=self.model,
                max_completion_tokens=2000,
                temperature=0.2
            )
            api_time = time.time() - start_time
            print(f"API request successful. Response received in {api_time:.2f} seconds.")
            return response.completion_message.content.text
        except Exception as e:
            api_time = time.time() - start_time
            print(f"Error after {api_time:.2f} seconds: {str(e)}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize with Llama API client
    classifier = VideoActionClassifier(
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        max_frames=8
    )
    
    # Analyze soccer video
    input_video_path = os.path.join(os.getcwd(), "videos", "oilers_hilights.mp4")
    result = classifier.analyze_video(
        video_path=input_video_path,
        actions=["goal", "pass", "shot", "save"],
        sport="hockey"
    )
    
    print("Analysis Results:")
    if isinstance(result, dict) and "error" in result:
        print(f"Error: {result['error']}")
    else:
        try:
            parsed_result = json.loads(result) if isinstance(result, str) else result
            print(json.dumps(parsed_result, indent=2))
        except:
            print(result)
