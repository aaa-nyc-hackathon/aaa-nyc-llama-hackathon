"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChevronLeft, ChevronRight, Search, ChevronDown, ChevronUp } from "lucide-react";
import Image from "next/image";
import { useState, useEffect } from "react";
import Link from "next/link";

// Define TypeScript interfaces for the analysis data
interface FeedbackDetails {
  final_conclusion: string;
  key_observations: string[];
  positives: string[];
  potential_issues: string[];
}

interface FeedbackItem {
  video_path: string;
  feedback: FeedbackDetails;
  start_frame: number;
  end_frame: number;
  marked_up_image_path?: string;
}

interface AnalysisResult {
  status: string;
  data: FeedbackItem[];
}

// Sub-component for individual collapsible feedback item
const CollapsibleFeedbackItem: React.FC<{ item: FeedbackItem; index: number }> = ({ item, index }) => {
  const [isOpen, setIsOpen] = useState(false);
  console.log("CollapsibleFeedbackItem received item:", item);

  return (
    <div className="mb-4 border border-gray-700 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center p-4 bg-gray-800 hover:bg-gray-700 focus:outline-none"
      >
        <span className="font-semibold text-lg">Feedback Item #{index + 1}</span>
        {isOpen ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
      </button>
      {isOpen && (
        <div className="p-4 bg-gray-900 text-sm">
          <h4 className="font-semibold mb-1 text-base">Video Segment File:</h4>
          <p className="mb-3 text-xs break-all">{item.video_path}</p>
          
          <h4 className="font-semibold mb-1 text-base">Conclusion:</h4>
          <p className="mb-3">{item.feedback.final_conclusion}</p>
          
          <h4 className="font-semibold mb-1 text-base">Key Observations:</h4>
          <ul className="list-disc list-inside mb-3 space-y-1">
            {item.feedback.key_observations.map((obs, i) => <li key={i}>{obs}</li>)}
          </ul>
          
          <h4 className="font-semibold mb-1 text-base">Positives:</h4>
          <ul className="list-disc list-inside mb-3 space-y-1">
            {item.feedback.positives.map((pos, i) => <li key={i}>{pos}</li>)}
          </ul>
          
          <h4 className="font-semibold mb-1 text-base">Potential Issues:</h4>
          <ul className="list-disc list-inside space-y-1">
            {item.feedback.potential_issues.map((iss, i) => <li key={i}>{iss}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
};

export default function Component() {
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const storedData = sessionStorage.getItem('analysisResultData');
    if (storedData) {
      try {
        const parsedData: AnalysisResult = JSON.parse(storedData);
        if (parsedData && parsedData.data && Array.isArray(parsedData.data)) {
          setAnalysisData(parsedData);
          console.log("Gallery: Analysis data loaded from session storage", parsedData);
          // Debug: Check the structure of the first item if it exists
          if (parsedData.data.length > 0) {
            console.log("First feedback item structure:", parsedData.data[0]);
            console.log("start_frame in first item:", parsedData.data[0].start_frame);
            console.log("end_frame in first item:", parsedData.data[0].end_frame);
          }
        } else {
          console.error("Gallery: Parsed data from session is not in expected format", parsedData);
          setError("Failed to load analysis data: Invalid format from session storage.");
        }
      } catch (e) {
        console.error("Gallery: Error parsing analysis data from sessionStorage", e);
        setError("Failed to load analysis data: Could not parse session storage.");
      }
    } else {
      console.log("Gallery: No analysis data found in sessionStorage.");
      // setError("No analysis data found. Please analyze a video first."); // Keep this commented unless no data is a hard error
    }
  }, []);

  // Logs to check state during render pass
  console.log("Gallery page rendering. Error state:", error);
  console.log("Gallery page rendering. AnalysisData state (raw):", analysisData);
  if (analysisData) {
    console.log("Gallery page rendering. AnalysisData.data.length:", analysisData.data?.length);
  }

  const displayVideoPath = analysisData?.data?.[0]?.video_path || "";

  const getFilename = (fullPath: string): string => {
    if (!fullPath) return '';
    const normalizedPath = fullPath.replace(/\\/g, '/'); // Normalize path separators
    const parts = normalizedPath.split('/');
    return parts[parts.length - 1];
  };

  const videoFilename = getFilename(displayVideoPath);
  // Assume videos will be placed in `public/processed_videos/` by the user
  const videoSrc = videoFilename ? `/processed_videos/${videoFilename}` : "";

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between bg-black text-white z-10 relative mx-auto px-6 sm:px-8 lg:px-12 py-4 w-full">
        <Link href="/landing" className="flex items-center gap-2">
          <img
            src="/logomark.png"
            alt="AthletIQ Logo"
            className="h-10 w-auto"
          />
          <span className="text-white text-xl font-bold">AthletIQ</span>
        </Link>

        <div className="relative max-w-xl w-full mx-4 hidden md:block">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            className="w-full rounded-full bg-gray-800/50 border border-gray-700 py-2 pl-10 pr-4 focus:outline-none focus:ring-1 focus:ring-gray-500"
            placeholder="Search Gallery (Not Implemented)"
          />
        </div>

        <Button
          variant="ghost"
          size="icon"
          className="md:hidden text-white hover:bg-gray-800"
          onClick={() => setIsSearchOpen(!isSearchOpen)}
        >
          <Search className="w-5 h-5" />
        </Button>

        <div className="flex items-center gap-6">
          <label
            htmlFor="video-upload-gallery"
            className="cursor-pointer bg-gradient-to-r from-[#D9202C] to-[#731117] text-white px-6 py-2 rounded-full font-medium transition-all duration-300 ease-in-out hover:brightness-110 hover:scale-105 shadow-md hover:shadow-lg"
          >
            New Analysis
            <input
              id="video-upload-gallery"
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.length) {
                  window.location.href = "/landing";
                }
              }}
            />
          </label>
        </div>
      </header>

      {isSearchOpen && (
        <div className="md:hidden px-3 pb-3 bg-black">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              placeholder="Search Gallery (Not Implemented)"
              className="w-full bg-gray-800 border-gray-700 text-white pl-10 rounded-full"
            />
          </div>
        </div>
      )}

      {/* Main Content Area - Adjusted for two columns */}
      <main className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Left Column: Video Display Area */}
        <div className="md:w-2/3 w-full h-[50vh] md:h-full bg-gray-900 flex flex-col items-center justify-center p-4 relative">
          <div className="text-center mb-4">
             <h2 className="text-xl font-semibold mb-2">Analyzed Video:</h2>
             <p className="text-sm text-gray-400 break-all">{displayVideoPath || "No video processed or path missing"}</p>
          </div>
          <div className="relative w-full max-w-2xl aspect-video bg-black">
            {videoSrc ? (
              <video controls src={videoSrc} className="w-full h-full object-contain">
                Your browser does not support the video tag.
                Attempting to load: {videoSrc}
              </video>
            ) : (
              <Image
                src="/basketball-game.png" // Fallback placeholder image
                alt="Video placeholder - video not found or path not specified"
                fill
                className="object-contain"
              />
            )}
          </div>
        </div>

        {/* Right Column: Feedback/Suggestions Area */}
        <div className="md:w-1/3 w-full md:h-full overflow-y-auto bg-gray-800 p-6">
          <h2 className="text-2xl font-bold mb-6 text-center">Analysis Feedback</h2>
          {error && <p className="text-red-400 bg-red-900 p-3 rounded-md">DEBUG: Error is: {error}</p>}
          
          {analysisData && analysisData.data && analysisData.data.length > 0 ? (
            analysisData.data.map((item, index) => (
              <CollapsibleFeedbackItem key={`${index}-${item.video_path}`} item={item} index={index} />
            ))
          ) : (
            <p className="text-gray-400">
              {error ? `Not rendering items due to error: ${error}` : "No analysis feedback items to display (analysisData might be null, or its data array is empty, or error state is active)."}
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
