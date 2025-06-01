"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

export default function Component() {
  const [apiStatus, setApiStatus] = useState<{ status: string; message: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  
  useEffect(() => {
    async function checkApiHealth() {
      try {
        // First attempt with default fetch
        try {
          const response = await fetch('http://localhost:8000/', {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
            },
            mode: 'cors'
          });
          
          const data = await response.json();
          console.log('API Health Check Response:', data);
          setApiStatus(data);
          return;
        } catch (initialErr) {
          console.warn('Initial fetch attempt failed, trying with no-cors:', initialErr);
        }
        
        // Second attempt with no-cors mode (will not be able to read the response)
        const fallbackResponse = await fetch('http://localhost:8000/', { 
          mode: 'no-cors' 
        });
        
        // If we get here, at least we know the API is running
        console.log('API is reachable but CORS is blocking proper access');
        setApiStatus({ status: 'online', message: 'API is running but CORS issues detected' });
        
      } catch (err) {
        console.error('Error checking API health:', err);
        setError('Failed to connect to API - Make sure backend is running');
      }
    }
    
    checkApiHealth();
  }, []);
  
  // Function to test the /api/analyze/ endpoint
  async function testAnalyzeEndpoint() {
    try {
      console.log('Sending request to analyze endpoint...');
      // First check if we can get the API status before sending the actual request
      if (!apiStatus || apiStatus.status !== 'online') {
        setError('API must be online to test the analyze endpoint');
        return;
      }
      
      if (!videoFile) {
        setError('Please select a video file first');
        return;
      }

      console.log('Uploading file:', videoFile.name);
      
      // Frontend upload code
      const formData = new FormData();
      formData.append("file", videoFile);

      const uploadRes = await fetch("http://localhost:8000/api/load", {
        method: "POST",
        body: formData
      });

      if (!uploadRes.ok) {
        const errorText = await uploadRes.text();
        console.error('Upload Error:', errorText);
        try {
          const errorData = JSON.parse(errorText);
          setError(`Upload Error: ${errorData.detail || 'Failed to upload file'}`);
        } catch (e) {
          setError(`Upload Error: ${uploadRes.status} ${uploadRes.statusText}`);
        }
        return;
      }
      
      const uploadData = await uploadRes.json();
      console.log('Upload successful:', uploadData);
      
      if (!uploadData.filepath) {
        setError('Upload response missing filepath');
        return;
      }

      // Analysis request
      const analysisRes = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: uploadData.filepath })
      });
      
      if (!analysisRes.ok) {
        try {
          const errorData = await analysisRes.json();
          console.error('API Error:', errorData);
          
          // Provide more helpful error message based on the error
          if (errorData.detail === "Analysis results not found") {
            setError(`API Error: The file was not found by the backend. 
                     Make sure it's in the current working directory of the backend server.`);
          } else {
            setError(`Analysis Error: ${errorData.detail || JSON.stringify(errorData)}`);
          }
        } catch (e) {
          setError(`Analysis Error: ${analysisRes.status} ${analysisRes.statusText}`);
        }
        return;
      }
      
      const data = await analysisRes.json();
      console.log('API Analyze Endpoint Response:', data);
      setAnalysisData(data);
      alert('Analysis data fetched successfully! Check console for details.');
    } catch (err: any) {
      console.error('Error fetching from analyze endpoint:', err);
      setError(`Failed to fetch analysis data: ${err.message || 'Unknown error'}`);
    }
  }
  
  return (
    <div className="min-h-screen bg-[#000000] text-white">
      {/* Navigation */}
      <nav className="flex items-center justify-between p-6">
        <Link href="/landing" className="flex items-center gap-2">
          <img
            src="/logomark.png"
            alt="AthletIQ Logo"
            className="h-10 w-auto"
          />
          <span className="text-white text-xl font-bold">AthletIQ</span>
        </Link>
        <div className="flex items-center gap-4">
          {apiStatus && (
            <div className="flex items-center gap-2 text-sm">
              <span className={`w-2 h-2 rounded-full ${apiStatus.status === 'online' ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-gray-300">API: {apiStatus.status}</span>
            </div>
          )}
          {error && (
            <div className="text-sm text-red-400">{error}</div>
          )}
          <Link href="/gallery" className="text-white text-lg">
            Login
          </Link>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex flex-col items-center justify-center min-h-[calc(100vh-120px)] px-6">
        <h1 className="text-[120px] md:text-[160px] lg:text-[200px] font-bold text-white text-center leading-none mb-8">
          AthletIQ
        </h1>

        <div className="flex flex-col items-center gap-4">
          <label
            htmlFor="video-upload"
            className="cursor-pointer bg-gradient-to-r from-[#D9202C] to-[#731117] transition-all duration-300 ease-in-out hover:brightness-110 hover:scale-105 shadow-md hover:shadow-lg text-white px-16 py-6 text-2xl font-bold rounded-full"
          >
            ANALYZE YOUR GAME
            <input
              id="video-upload"
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.length) {
                  setVideoFile(e.target.files[0]);
                  // Uncomment this line if you want to navigate to loading page immediately
                  // window.location.href = '/loading';
                }
              }}
            />
          </label>
          
          {/* Test button for the /api/analyze/ endpoint */}
          <button 
            onClick={testAnalyzeEndpoint}
            className="mt-4 bg-blue-600 hover:bg-blue-700 transition-colors px-8 py-3 rounded-lg text-white font-medium"
          >
            Test Analyze API Endpoint
          </button>
          
          {analysisData && (
            <div className="mt-4 p-4 bg-gray-900 rounded-md max-w-lg mx-auto overflow-auto">
              <p className="text-green-400 mb-2">Analysis data received! Check console for full details.</p>
              <p className="text-gray-400 text-sm">
                {analysisData.data ? `Found ${analysisData.data.length} feedback items` : 'No data found'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
