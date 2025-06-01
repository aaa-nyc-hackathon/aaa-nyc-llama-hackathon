"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

export default function Component() {
  const [apiStatus, setApiStatus] = useState<{ status: string; message: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  
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
                window.location.href = '/loading';
              }
            }}
          />
        </label>
      </div>
    </div>
  );
}
