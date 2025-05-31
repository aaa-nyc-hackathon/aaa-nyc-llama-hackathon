"""
Script to run the FastAPI server
"""
import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables from .env file if present
load_dotenv()


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Define the request body using Pydantic
class LoadBody(BaseModel):
    file_path: str

# Define the POST route
@app.post("/load/")
async def load(body: LoadBody):
    print(body.file_path)
    if not (os.path.exists(body.file_path) or os.path.exists(os.path.join(os.getcwd(), body.file_path))):
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "message": "File successfully loaded"
    }

if __name__ == "__main__":
    uvicorn.run("run_server:app", host="0.0.0.0", port=8000, reload=True)

