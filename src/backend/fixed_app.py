#from fastapi import FastAPI, UploadFile, File, HTTPException
#from fastapi.middleware.cors import CORSMiddleware
#import os
#import tempfile
#import uuid
#from typing import List, Dict, Any
#import json
#from pydantic import BaseModel
#
#app = FastAPI(title="Video Analysis API")
#
## Add CORS middleware to allow cross-origin requests
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["http://localhost:3000", "*"],  # Allow requests from Next.js dev server
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)
#
#@app.get("/")
#async def root():
#    """Health check endpoint"""
#    return {"status": "online", "message": "Video Analysis API is running"}
#
#class LoadBody(BaseModel):
#    file_path: str
#
#@app.post("/api/load/")
#async def load(body: LoadBody):
#    print(body.file_path)
#    if not (os.path.exists(body.file_path) or os.path.exists(os.path.join(os.getcwd(), body.file_path))):
#        raise HTTPException(status_code=404, detail="File not found")
#    
#    return {
#        "message": "File successfully loaded"
#    }
#
#@app.post("/api/analyze/")
#async def analyze(body: LoadBody):
#    """
#    Analyze the uploaded video file and return actions with timestamps.
#    
#    Args:
#        file: The video file to analyze.
#        
#    Returns:
#        A dictionary containing analysis results.
#    """
#    # load in "final_output.json" from the current directory
#    final_output_path = os.path.join(os.getcwd(), "final_output.json")
#    if not os.path.exists(final_output_path):
#        raise HTTPException(status_code=404, detail="Analysis results not found")
#    with open(final_output_path, 'r') as f:
#        analysis_results = json.loads(f)
#    
#    output_arr = []
#    for video_segment in analysis_results["video_segments"]:
#        video_path = video_segment["video_segment_path"]
#        for info in video_segment["list_of_info"]:
#            for team in info["team"]:
#                for player in team["obj"]:
#                    for frame in player["frames"]:
#                        if frame["feedback"] is not None:
#                            output_arr.append({
#                                "video_path": video_path,
#                                "feedback": frame["feedback"],
#                            })
#    return {"data": output_arr}
#
#
#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)
#

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
from typing import List, Dict, Any
import json
from pydantic import BaseModel

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
    """Handle file upload and return absolute path""" 
    try:
        # Create upload directory if missing
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_id = uuid.uuid4().hex
        saved_name = f"{unique_id}{file_ext}"
        file_path = os.path.abspath(os.path.join(upload_dir, saved_name))

        # Save file to disk
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024):
                buffer.write(content)
        
        return {
            "status": "success",
            "filepath": file_path
        }
    
    except Exception as e:
        raise HTTPException(500, f"File processing failed: {str(e)}")

@app.post("/api/analyze")
async def analyze(body: LoadBody):
    """Return received file path for verification"""
    if not os.path.exists(body.file_path):
        raise HTTPException(404, "File not found at specified path")
    
    return {
        "status": "success",
        "analyzed_path": body.file_path
    }

@app.get("/")
async def health_check():
    return {"status": "online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
