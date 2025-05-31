from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
from typing import List, Dict, Any
import json

app = FastAPI(title="Video Analysis API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Video Analysis API is running"}

@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload and analyze a video file.
    
    This endpoint accepts a video file, processes it, and returns timestamps with actions.
    """
    # Check file extension rather than content-type which might not be set correctly
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"File must be a video with one of these extensions: {', '.join(valid_extensions)}")
    
    # Log information about the uploaded file
    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
    
    try:
        # Create temp file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            temp_path = temp.name
            # Write uploaded file to disk
            content = await file.read()
            temp.write(content)
        
        # In a real implementation, you would process the video here
        # For now, return dummy data as requested
        
        # Generate dummy timestamps and actions
        dummy_results = generate_dummy_results()
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
        return dummy_results
        
    except Exception as e:
        # Make sure to clean up temp file in case of errors
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

def generate_dummy_results() -> Dict[str, Any]:
    """Generate dummy analysis results with timestamps and actions"""
    return {
        "video_id": str(uuid.uuid4()),
        "analysis_timestamp": "2025-05-31T12:00:00Z",
        "actions": [
            {
                "timestamp": "00:00:05",
                "action_type": "event",
                "confidence": 0.95,
                "description": "example"
            },
            {
                "timestamp": "00:00:15",
                "action_type": "highlight",
                "confidence": 0.87,
                "description": "example"
            },
            {
                "timestamp": "00:00:30",
                "action_type": "play",
                "confidence": 0.92,
                "description": "example"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
