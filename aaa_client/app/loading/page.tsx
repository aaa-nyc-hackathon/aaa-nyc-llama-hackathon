"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

// Define a type for the expected structure of progress_steps items
interface ProgressStep {
  percentage_start: number;
  percentage_end: number;
  description: string;
  time_start_seconds: number;
  time_end_seconds: number;
}

// Define a type for the data expected from sessionStorage
interface LoadingData {
  filepath: string;
  total_estimated_time_seconds: number;
  progress_steps: ProgressStep[];
  // Add any other fields that might be present in uploadData
  status?: string;
}

export default function Component() {
  const [progress, setProgress] = useState(0);
  const [currentStageMessage, setCurrentStageMessage] = useState("Initializing...");
  const [error, setError] = useState<string | null>(null);
  const [analysisComplete, setAnalysisComplete] = useState(false); // New state for API call completion
  const router = useRouter();

  // Refs to store data that doesn't need to trigger re-renders on change
  const loadingDataRef = useRef<LoadingData | null>(null);
  const analysisCalledRef = useRef(false); // To prevent multiple API calls
  const currentStageIndexRef = useRef(0); // To track current step in progress_steps
  const stageStartTimeRef = useRef(0); // To track start time of the current real-time progress stage
  const intervalIdRef = useRef<NodeJS.Timeout | null>(null); // Ref to store interval ID

  useEffect(() => {
    const data = sessionStorage.getItem('loadingData');
    if (data) {
      try {
        const parsedData: LoadingData = JSON.parse(data);
        if (!parsedData.filepath || !parsedData.progress_steps || parsedData.progress_steps.length === 0 || typeof parsedData.total_estimated_time_seconds === 'undefined') {
          setError("Essential loading data is missing or invalid. Redirecting to landing.");
          console.error("Invalid loading data:", parsedData);
          setTimeout(() => router.push("/landing"), 3000);
          return;
        }
        loadingDataRef.current = parsedData;
        console.log("Loading data retrieved:", parsedData);
        setCurrentStageMessage(parsedData.progress_steps[0]?.description || "Preparing analysis...");
        stageStartTimeRef.current = Date.now(); // Mark the start of the overall process simulation
      } catch (e) {
        console.error("Failed to parse loading data from sessionStorage:", e);
        setError("Failed to retrieve loading information. Redirecting to landing.");
        setTimeout(() => router.push("/landing"), 3000);
        return;
      }
    } else {
      setError("No loading data found. Redirecting to landing page.");
      console.warn("sessionStorage did not contain 'loadingData'.");
      setTimeout(() => router.push("/landing"), 3000); // Redirect if no data
      return;
    }

    // Cleanup sessionStorage item if desired, though it will be overwritten or cleared on new upload
    // sessionStorage.removeItem('loadingData'); // Optional: clear it once read
  }, [router]);

  useEffect(() => {
    if (!loadingDataRef.current) return;

    // Initiate API call as soon as filepath is available and not already called
    if (loadingDataRef.current.filepath && !analysisCalledRef.current) {
      analysisCalledRef.current = true; // Prevent multiple calls
      callAnalyzeApi(loadingDataRef.current.filepath);
    }

    const { progress_steps, total_estimated_time_seconds } = loadingDataRef.current;
    
    // If there are no steps, or total time is zero, something is wrong.
    if (progress_steps.length === 0 || total_estimated_time_seconds <= 0) {
        console.warn("No progress steps or zero total time, halting progress updates.");
        // Optionally set error or redirect
        return;
    }

    const overallStartTime = stageStartTimeRef.current; // Start time of the whole process

    const interval = setInterval(() => {
        if (!loadingDataRef.current || analysisComplete) { // Stop if analysis is already complete
            clearInterval(interval);
            return;
        }
        const elapsedTimeMs = Date.now() - overallStartTime;
        const elapsedTimeSec = elapsedTimeMs / 1000;

        let currentGlobalPercentage = (elapsedTimeSec / total_estimated_time_seconds) * 100;
        currentGlobalPercentage = Math.min(currentGlobalPercentage, 100); // Cap at 100%

        setProgress(currentGlobalPercentage);

        // Find the current stage based on real-time progress
        let activeStageDescription = "Processing...";
        if (currentGlobalPercentage < 100) {
            let currentStep = progress_steps[progress_steps.length - 1]; // Default to last step description
            for (const step of progress_steps) {
                if (currentGlobalPercentage < step.percentage_end) {
                    currentStep = step;
                    break;
                }
            }
            activeStageDescription = currentStep.description;
        } else {
            // Animation is complete
            activeStageDescription = analysisComplete ? "Analysis Complete! Preparing results..." : "Finalizing analysis, awaiting server response...";
        }
        setCurrentStageMessage(activeStageDescription);

        if (currentGlobalPercentage >= 100) {
          setProgress(100);
          clearInterval(interval); 
          // Animation finished, actual API call might still be in progress or already done.
        }
    }, 200);

    intervalIdRef.current = interval; // Store interval ID

    return () => {
        if (intervalIdRef.current) clearInterval(intervalIdRef.current);
    };
  }, [analysisComplete]); // Rerun if analysisComplete changes to update message, also runs on initial mount with loadingDataRef.current

  async function callAnalyzeApi(filepath: string) {
    console.log(`Calling /api/analyze for filepath: ${filepath}`);
    // Message will be updated by the progress interval, or if API finishes first
    try {
      const analysisRes = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: filepath })
      });

      if (!analysisRes.ok) {
        const errorData = await analysisRes.json().catch(() => ({ detail: "Failed to parse error from analysis API." }));
        console.error('Analysis API Error:', errorData);
        setError(`Analysis Failed: ${errorData.detail || analysisRes.statusText}`);
        setAnalysisComplete(false); // Explicitly set to false on error
        return;
      }

      const data = await analysisRes.json();
      console.log('API Analyze Endpoint Response:', data);
      
      // Log specific player data to verify it's coming from the API
      if (data.data && Array.isArray(data.data)) {
        console.log('Total feedback items:', data.data.length);
        data.data.forEach((item: any, index: number) => {
          if (item.player) {
            console.log(`Feedback item ${index}: Player #${item.player.jersey_number}, Time: ${item.start_frame}s-${item.end_frame}s`);
          } else {
            console.log(`Feedback item ${index}: No player data found`);
          }
        });
      }
      
      sessionStorage.setItem('analysisResultData', JSON.stringify(data)); 
      setAnalysisComplete(true); // Signal API success
      // Message will be updated by progress interval or the redirect effect

      // If API finishes before animation, stop animation and set to 100%
      if (intervalIdRef.current) {
        clearInterval(intervalIdRef.current);
        intervalIdRef.current = null; // Clear the ref
      }
      setProgress(100);
      setCurrentStageMessage("Analysis processing complete! Finalizing...");

    } catch (err: any) {
      console.error('Error calling analyze endpoint:', err);
      setError(`Failed to get analysis results: ${err.message || 'Unknown error'}`);
      setAnalysisComplete(false); // Explicitly set to false on error
    }
  }

  // Effect for redirection when both animation and API call are complete
  useEffect(() => {
    if (progress >= 100 && analysisComplete) {
      setCurrentStageMessage("Analysis Complete! Redirecting...");
      setTimeout(() => {
        router.push("/analytics");
      }, 1500); 
    }
  }, [progress, analysisComplete, router]);

  return (
    <div className="min-h-screen bg-[#000000] text-white">
      {/* Header Navigation */}
      <header className="flex items-center justify-between p-6">
        {/* Logo */}
        <Link href="/landing" className="flex items-center gap-2">
          <img
            src="/logomark.png"
            alt="AthletIQ Logo"
            className="h-10 w-auto"
          />
          <span className="text-white text-xl font-bold">AthletIQ</span>
        </Link>

        {/* Right Navigation */}
        <div className="flex items-center gap-4">
          <Link href="/gallery" className="text-white text-lg">
            Gallery
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main
        className="flex flex-col items-center justify-center flex-1 px-6"
        style={{ minHeight: "calc(100vh - 120px)" }}
      >
        {error ? (
          <div className="text-center">
            <h2 className="text-2xl font-medium text-red-500 mb-4">Error</h2>
            <p className="text-gray-300 mb-6">{error}</p>
            <button 
              onClick={() => router.push('/landing')}
              className="bg-blue-600 hover:bg-blue-700 transition-colors px-8 py-3 rounded-lg text-white font-medium"
            >
              Try Again
            </button>
          </div>
        ) : (
          <>
            <div className="w-full max-w-lg mb-8">
              <div className="h-12 bg-[#060606] rounded-full overflow-hidden border border-[#494949] relative">
                <div
                  className="h-full bg-gradient-to-r from-red-600 to-red-500 rounded-full transition-all duration-300 ease-out relative"
                  style={{ width: `${progress}%` }}
                >
                  {/* Animated shine effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse"></div>
                </div>
              </div>

              {/* Progress percentage */}
              <div className="flex justify-between items-center mt-2">
                <span className="text-[#757575] text-sm">
                  {Math.round(progress)}%
                </span>
                <span className="text-[#757575] text-sm">
                  {progress < 100 ? "Processing..." : (analysisComplete ? "Complete" : "Finalizing...")}
                </span>
              </div>
            </div>

            {/* Status Text */}
            <h2 className="text-2xl font-medium text-[#ffffff] text-center transition-all duration-500 ease-in-out">
              {currentStageMessage}
            </h2>

            {/* Loading indicator dots */}
            {progress < 100 && !error && (
              <div className="flex gap-1 mt-4">
                <div className="w-2 h-2 bg-red-600 rounded-full animate-bounce"></div>
                <div
                  className="w-2 h-2 bg-red-600 rounded-full animate-bounce"
                  style={{ animationDelay: "0.1s" }}
                ></div>
                <div
                  className="w-2 h-2 bg-red-600 rounded-full animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                ></div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

