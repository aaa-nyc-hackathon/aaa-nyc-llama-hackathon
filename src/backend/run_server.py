"""
Script to run the FastAPI server
"""
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
