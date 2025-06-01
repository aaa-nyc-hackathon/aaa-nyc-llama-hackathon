from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
from typing import List, Dict, Any
import json
from pydantic import BaseModel
import cv2 # Added for video duration
import time
import shutil # Added for file copying
from workflow import start_workflow

app = FastAPI(title="Video Analysis API")

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoadBody(BaseModel):
    file_path: str

@app.post("/api/load")
async def load(file: UploadFile = File(...)):
    """Handle file upload, determine duration, calculate processing ETA, and return details"""
    try:
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_ext = os.path.splitext(file.filename)[1]
        unique_id = uuid.uuid4().hex
        saved_name = f"{unique_id}{file_ext}"
        file_path = os.path.abspath(os.path.join(upload_dir, saved_name))

        with open(file_path, "wb") as buffer:
            while content := await file.read(1024):
                buffer.write(content)
        
        # Determine video duration using OpenCV
        video_duration_seconds = 0
        cap = None
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise HTTPException(500, "Could not open video file to determine duration.")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps > 0 and frame_count > 0:
                video_duration_seconds = frame_count / fps
            else:
                raise HTTPException(500, "Could not read video properties (FPS, Frame Count) to determine duration.")
        except Exception as e:
            # If duration can't be determined, could fall back to a default or raise error
            # For now, re-raising as an HTTP exception to make it clear
            raise HTTPException(status_code=500, detail=f"Error determining video duration: {str(e)}")
        finally:
            if cap:
                cap.release()

        if video_duration_seconds <= 0:
            raise HTTPException(400, "Video duration must be positive. Calculated duration was zero or negative.")

        # Calculate ETA and progress steps
        base_video_duration_seconds = 30.0
        base_processing_time_seconds = 8 * 60.0

        total_estimated_time_seconds = (video_duration_seconds / base_video_duration_seconds) * base_processing_time_seconds

        progress_steps = [
            {
                "percentage_start": 0, "percentage_end": 30,
                "description": "extracting player locations with llama 4",
                "time_start_seconds": 0,
                "time_end_seconds": round(total_estimated_time_seconds * 0.30)
            },
            {
                "percentage_start": 30, "percentage_end": 40,
                "description": "using opencv to calculate player motions",
                "time_start_seconds": round(total_estimated_time_seconds * 0.30),
                "time_end_seconds": round(total_estimated_time_seconds * 0.40)
            },
            {
                "percentage_start": 40, "percentage_end": 100,
                "description": "analyzing player footage and providing feedback with llama 4",
                "time_start_seconds": round(total_estimated_time_seconds * 0.40),
                "time_end_seconds": round(total_estimated_time_seconds)
            }
        ]
        
        return {
            "status": "success",
            "filepath": file_path,
            "total_estimated_time_seconds": round(total_estimated_time_seconds),
            "progress_steps": progress_steps
        }
    
    except HTTPException as he: # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        raise HTTPException(500, f"File processing failed: {str(e)}")

@app.post("/api/analyze")
async def analyze(body: LoadBody):
    """Process video, copy segments to public folder, and return analysis with updated paths."""
    if not os.path.exists(body.file_path):
        raise HTTPException(404, "Original uploaded file for analysis not found at specified path: " + body.file_path)
    
    try:
        # Define the target directory for frontend videos
        # Assumes fixed_app.py is in src/backend/
        # Path to workspace root: ../../ from os.getcwd() if CWD is src/backend/
        # then aaa_client/public/processed_videos/
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_script_dir, "..", "..")) # Up to aaa-nyc-llama-hackathon
        frontend_target_dir = os.path.join(project_root, "aaa_client", "public", "processed_videos")
        
        os.makedirs(frontend_target_dir, exist_ok=True)
        print(f"Ensured frontend video directory exists: {frontend_target_dir}")

        #starting real workflow 
        analysis_results = start_workflow(body.file_path)

        #json_path = os.path.join(project_root, "src", "backend", "final_output.json")
        #print(f"Looking for JSON file at: {json_path}")
        #
        #if not os.path.exists(json_path):
        #    print(f"JSON file not found at {json_path}")
        #    return {"status": "error", "message": "Analysis results file (final_output.json) not found"}
        #    
        #with open(json_path, 'r') as f:
        #    analysis_results = json.load(f)
        
        output_arr = []
        
        if "video_segments" not in analysis_results:
            return {"status": "error", "message": "Invalid JSON structure: missing 'video_segments' key"}
            
        for video_segment in analysis_results["video_segments"]:
            source_video_segment_path = video_segment["video_segment_path"]
            
            if not os.path.exists(source_video_segment_path):
                print(f"Warning: Source video segment not found: {source_video_segment_path}. Skipping this segment.")
                # Optionally include this info in the response or error out
                continue 

            segment_filename = os.path.basename(source_video_segment_path)
            destination_public_path = os.path.join(frontend_target_dir, segment_filename)

            try:
                shutil.copy2(source_video_segment_path, destination_public_path)
                print(f"Copied: {source_video_segment_path} TO {destination_public_path}")
            except Exception as copy_e:
                print(f"Error copying {segment_filename} to public dir: {copy_e}")
                # Decide if you want to error out or just skip this file and continue
                # For now, we'll use original path if copy fails, but ideally, this should be robust
                # or the frontend should be informed that the public URL is not available.
                # To keep frontend logic simple, we'll still return just the filename, 
                # but log the error. Frontend will get a 404 for this specific video.

            # Process feedback items for this segment
            for info in video_segment.get("list_of_info", []):
                if isinstance(info.get("team"), str):
                    for player in info.get("obj", []):
                        jersey_number = player["jersey_number"]
                        for frame in player.get("frames", []):
                            if frame.get("feedback") is not None:
                                output_arr.append({
                                    "video_path": segment_filename, # Use just the filename
                                    "feedback": frame["feedback"],
                                    "start_frame": frame["start_frame"],
                                    "end_frame": frame["end_frame"],
                                    "x": frame["x"],
                                    "y": frame["y"],
                                    "player" : jersey_number
                                })
                elif isinstance(info.get("team"), list):
                    for team_info in info["team"]:
                        for player in team_info.get("obj", []):
                            print(f"player>>>>{player}")
                            jersey_number = player["jersey_number"]
                            for frame in player.get("frames", []):
                                if frame.get("feedback") is not None:
                                    output_arr.append({
                                        "video_path": segment_filename, # Use just the filename
                                        "feedback": frame["feedback"],
                                        "start_frame": frame["start_frame"],
                                        "end_frame": frame["end_frame"],
                                        "x": frame["x"],
                                        "y": frame["y"],
                                        "player" : jersey_number
                                    })
                                
        
        if not output_arr:
            return {"status": "warning", "message": "No feedback items extracted from analysis results", "data": []}
        return {"status": "success", "data": output_arr}

    except HTTPException as he:
        raise he # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Error processing analysis results: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Error processing analysis results: {str(e)}"}

@app.get("/")
async def health_check():
    return {"status": "online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
