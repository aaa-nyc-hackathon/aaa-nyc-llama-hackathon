"use client";

import { Search, ChevronDown, ChevronUp, PlayCircle } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";

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
  x?: number;
  y?: number;
  player?: number;
}

interface AnalysisResult {
  status: string;
  data: FeedbackItem[];
}

// Sub-component for individual collapsible feedback item
const CollapsibleFeedbackItem: React.FC<{
  item: FeedbackItem;
  index: number;
  onSelectSegment: (videoPath: string) => void;
  isActiveSegment: boolean;
  onSeekToTimestamp: (startTime: number, endTime?: number) => void;
}> = ({ item, index, onSelectSegment, isActiveSegment, onSeekToTimestamp }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div
      className={`mb-4 border rounded-lg overflow-hidden bg-gray-800 ${
        isActiveSegment ? "border-red-500 shadow-lg" : "border-gray-700"
      }`}
    >
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex justify-between items-center p-3 sm:p-4 bg-gray-700 hover:bg-gray-600 focus:outline-none text-left"
      >
        <span className="font-semibold text-base sm:text-lg">
          {item.player !== undefined
            ? `Player #${item.player} - ${item.start_frame}s-${item.end_frame}s`
            : `Feedback #${index + 1}`}
        </span>
        {isOpen ? (
          <ChevronUp className="w-5 h-5" />
        ) : (
          <ChevronDown className="w-5 h-5" />
        )}
      </button>
      {isOpen && (
        <div className="p-3 sm:p-4 bg-gray-800 text-xs sm:text-sm">
          <div className="flex justify-between items-center mb-2">
            <h4 className="font-semibold text-sm sm:text-base">
              Video Segment:
            </h4>
            {!isActiveSegment && (
              <button
                onClick={() => onSelectSegment(item.video_path)}
                className="flex items-center text-red-400 hover:text-red-300 text-xs sm:text-sm py-1 px-2 rounded bg-gray-700 hover:bg-gray-600 transition-colors"
              >
                <PlayCircle className="w-4 h-4 mr-1" /> View this Segment
              </button>
            )}
          </div>
          <p className="mb-2 sm:mb-3 text-xs break-all italic text-gray-400">
            {item.video_path}
          </p>

          {item.player !== undefined && (
            <>
              <h4 className="font-semibold mb-1 text-sm sm:text-base">
                Player:
              </h4>
              <p className="mb-2 sm:mb-3">Jersey Number #{item.player}</p>
            </>
          )}

          {item.start_frame !== undefined && item.end_frame !== undefined && (
            <>
              <h4 className="font-semibold mb-1 text-sm sm:text-base">
                Time Range:
              </h4>
              <button
                onClick={() => {
                  if (!isActiveSegment) {
                    onSelectSegment(item.video_path);
                  }
                  setTimeout(
                    () => onSeekToTimestamp(item.start_frame, item.end_frame),
                    100
                  );
                }}
                className="mb-2 sm:mb-3 text-blue-400 hover:text-blue-300 underline cursor-pointer text-left transition-colors flex items-center gap-1"
              >
                <PlayCircle className="w-4 h-4" />
                {item.start_frame} seconds to {item.end_frame} seconds
              </button>
            </>
          )}

          <h4 className="font-semibold mb-1 text-sm sm:text-base">
            Conclusion:
          </h4>
          <p className="mb-2 sm:mb-3">{item.feedback.final_conclusion}</p>

          <h4 className="font-semibold mb-1 text-sm sm:text-base">
            Key Observations:
          </h4>
          <ul className="list-disc list-inside mb-2 sm:mb-3 space-y-1">
            {item.feedback.key_observations.map((obs, i) => (
              <li key={`obs-${index}-${i}`}>{obs}</li>
            ))}
          </ul>

          <h4 className="font-semibold mb-1 text-sm sm:text-base">
            Positives:
          </h4>
          <ul className="list-disc list-inside mb-2 sm:mb-3 space-y-1">
            {item.feedback.positives.map((pos, i) => (
              <li key={`pos-${index}-${i}`}>{pos}</li>
            ))}
          </ul>

          <h4 className="font-semibold mb-1 text-sm sm:text-base">
            Potential Issues:
          </h4>
          <ul className="list-disc list-inside space-y-1">
            {item.feedback.potential_issues.map((iss, i) => (
              <li key={`iss-${index}-${i}`}>{iss}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

const getFilename = (fullPath: string): string => {
  if (!fullPath) return "";
  const normalizedPath = fullPath.replace(/\\/g, "/"); // Normalize path separators
  const parts = normalizedPath.split("/");
  return parts[parts.length - 1];
};

export default function GameAnalysisPage() {
  const router = useRouter();
  const [slideOut, setSlideOut] = useState(false);
  const [analysisReport, setAnalysisReport] = useState<AnalysisResult | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const [activeVideoSrc, setActiveVideoSrc] = useState<string>("");
  const [activeVideoPath, setActiveVideoPath] = useState<string>(""); // To track original path for isActiveSegment
  const [videoRef, setVideoRef] = useState<HTMLVideoElement | null>(null);
  const [activeSegment, setActiveSegment] = useState<{
    start: number;
    end: number;
  } | null>(null);
  const [playbackRate, setPlaybackRate] = useState<number>(0.5); // Default to 0.5x speed
  const [isLooping, setIsLooping] = useState<boolean>(true);
  const [selectedPlayer, setSelectedPlayer] = useState<number | null>(null); // Selected player filter
  const [availablePlayers, setAvailablePlayers] = useState<number[]>([]); // All unique player numbers

  useEffect(() => {
    const storedData = sessionStorage.getItem("analysisResultData");
    if (storedData) {
      try {
        const parsedData: AnalysisResult = JSON.parse(storedData);
        if (
          parsedData &&
          parsedData.data &&
          Array.isArray(parsedData.data) &&
          parsedData.data.length > 0
        ) {
          // Sort data by player jersey number
          const sortedData = [...parsedData.data].sort((a, b) => {
            const playerA = a.player ?? Infinity;
            const playerB = b.player ?? Infinity;
            return playerA - playerB;
          });

          // Extract unique player numbers
          const uniquePlayers = [
            ...new Set(
              sortedData
                .filter((item) => item.player !== undefined)
                .map((item) => item.player!)
            ),
          ].sort((a, b) => a - b);

          setAvailablePlayers(uniquePlayers);
          setAnalysisReport({ ...parsedData, data: sortedData });
          console.log(
            "Analytics: Analysis data loaded and sorted by player",
            sortedData
          );
          console.log("Available players:", uniquePlayers);

          // Log player data to verify it's being received
          console.log("Analytics: Checking player data in received items:");
          sortedData.forEach((item, index) => {
            if (item.player !== undefined) {
              console.log(
                `Item ${index}: Player #${item.player}, Time: ${item.start_frame}s-${item.end_frame}s`
              );
            } else {
              console.log(`Item ${index}: No player data`);
            }
          });

          // Initialize with the first segment
          const firstSegmentPath = sortedData[0].video_path;
          setActiveVideoPath(firstSegmentPath);
          const firstSegmentFilename = getFilename(firstSegmentPath);
          if (firstSegmentFilename) {
            setActiveVideoSrc(`/processed_videos/${firstSegmentFilename}`);
          }
        } else {
          const errMsg =
            parsedData?.data?.length === 0
              ? "No feedback data found in the analysis results."
              : "Failed to load analysis details: Invalid or empty format.";
          console.error(
            "Analytics: Parsed data is not in expected format or empty",
            parsedData
          );
          setError(errMsg);
        }
      } catch (e) {
        console.error(
          "Analytics: Error parsing analysis data from sessionStorage",
          e
        );
        setError("Failed to load analysis details: Could not parse.");
      }
    } else {
      console.log("Analytics: No analysis data found in sessionStorage.");
      setError("No analysis details found. Please process a video first.");
    }
  }, []);

  const handleSelectSegment = (videoPath: string) => {
    const filename = getFilename(videoPath);
    if (filename) {
      setActiveVideoPath(videoPath); // Store the original path
      setActiveVideoSrc(`/processed_videos/${filename}`);
    } else {
      console.warn(
        "Could not select segment: filename is empty for path",
        videoPath
      );
      setActiveVideoSrc(""); // Clear or set to a placeholder/error state
      setActiveVideoPath("");
    }
  };

  const pageTitle = activeVideoSrc
    ? `Analysis: ${getFilename(activeVideoPath)}`
    : "Game Analysis";
  const feedbackCount = analysisReport?.data?.length || 0;

  // Filter data based on selected player
  const filteredData =
    analysisReport?.data?.filter((item) => {
      if (selectedPlayer === null) return true;
      return item.player === selectedPlayer;
    }) || [];

  const displayedFeedbackCount = filteredData.length;

  // Function to seek video to specific timestamp
  const seekToTimestamp = (startTime: number, endTime?: number) => {
    if (videoRef) {
      videoRef.currentTime = startTime;
      videoRef.playbackRate = playbackRate;

      if (endTime !== undefined) {
        setActiveSegment({ start: startTime, end: endTime });
      } else {
        setActiveSegment(null);
      }

      videoRef.play().catch((e) => console.log("Video play failed:", e));
    }
  };

  // Handle video timeupdate for looping
  useEffect(() => {
    if (!videoRef || !activeSegment || !isLooping) return;

    const handleTimeUpdate = () => {
      if (videoRef.currentTime >= activeSegment.end) {
        videoRef.currentTime = activeSegment.start;
      }
    };

    videoRef.addEventListener("timeupdate", handleTimeUpdate);
    return () => videoRef.removeEventListener("timeupdate", handleTimeUpdate);
  }, [videoRef, activeSegment, isLooping]);

  // Update playback rate when it changes
  useEffect(() => {
    if (videoRef) {
      videoRef.playbackRate = playbackRate;
    }
  }, [videoRef, playbackRate]);

  return (
    <>
      <header className="flex items-center justify-between mb-8 bg-black text-white z-10 relative mx-auto px-6 sm:px-8 lg:px-12 py-4">
        <Link href="/" className="flex items-center gap-2">
          <img
            src="/logomark.png"
            alt="AthletIQ Logo"
            className="h-10 w-auto"
          />
          <span className="text-white text-xl font-bold">AthletIQ</span>
        </Link>

        <div className="relative max-w-xl w-full mx-4 hidden md:block">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
          <input
            type="text"
            className="w-full rounded-full bg-gray-800/50 border border-gray-700 py-2 pl-10 pr-4 focus:outline-none focus:ring-1 focus:ring-gray-500"
            placeholder="Search Analytics (Not Implemented)"
          />
        </div>

        <div className="flex items-center gap-6">
          <label
            htmlFor="video-upload-analytics"
            className="cursor-pointer bg-gradient-to-r from-[#D9202C] to-[#731117] text-white px-6 py-2 rounded-full font-medium transition-all duration-300 ease-in-out hover:brightness-110 hover:scale-105 shadow-md hover:shadow-lg"
          >
            New Analysis
            <input
              id="video-upload-analytics"
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.length) {
                  window.location.href = "/";
                }
              }}
            />
          </label>
          <Link href="/gallery" className="text-white text-lg">
            Gallery
          </Link>
        </div>
      </header>

      <motion.div
        initial={{ y: "100%", opacity: 0 }}
        animate={slideOut ? { y: "100%", opacity: 0 } : { y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: "easeInOut" }}
        onAnimationComplete={() => {
          if (slideOut) router.push("/gallery");
        }}
        className="min-h-screen bg-black text-white"
      >
        <main className="mx-auto px-6 sm:px-8 lg:px-12 py-4 pb-8">
          {error && (
            <div className="text-center py-10">
              <p className="text-red-500 text-xl">{error}</p>
              <Link
                href="/"
                className="mt-4 inline-block bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
              >
                Start New Analysis
              </Link>
            </div>
          )}
          {!error && analysisReport && (
            <>
              <div className="flex justify-between items-center mb-6 flex-wrap">
                <h1
                  className="text-2xl sm:text-3xl lg:text-4xl font-bold truncate mr-4"
                  title={pageTitle}
                >
                  {pageTitle}
                </h1>
                {displayedFeedbackCount > 0 && (
                  <span className="text-lg sm:text-xl lg:text-2xl font-semibold whitespace-nowrap">
                    {displayedFeedbackCount} Feedback points
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 sm:gap-8">
                <div className="xl:col-span-2 space-y-4">
                  {/* Playback Controls */}
                  <div className="flex items-center justify-between bg-gray-800 rounded-md p-3">
                    <div className="flex items-center gap-4">
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={isLooping}
                          onChange={(e) => setIsLooping(e.target.checked)}
                          className="w-4 h-4"
                        />
                        <span className="text-sm">Loop Segment</span>
                      </label>

                      <div className="flex items-center gap-2">
                        <span className="text-sm">Speed:</span>
                        <select
                          value={playbackRate}
                          onChange={(e) =>
                            setPlaybackRate(parseFloat(e.target.value))
                          }
                          className="bg-gray-700 text-white px-2 py-1 rounded text-sm"
                        >
                          <option value="0.25">0.25x</option>
                          <option value="0.5">0.5x</option>
                          <option value="0.75">0.75x</option>
                          <option value="1">1x</option>
                        </select>
                      </div>
                    </div>

                    {activeSegment && (
                      <button
                        onClick={() => {
                          setActiveSegment(null);
                          if (videoRef) videoRef.playbackRate = 1;
                        }}
                        className="text-sm text-gray-400 hover:text-white transition-colors"
                      >
                        Clear Loop
                      </button>
                    )}
                  </div>

                  <div className="relative aspect-video w-full bg-black rounded-md overflow-hidden border border-gray-700">
                    {activeSegment && (
                      <div className="absolute top-2 left-2 z-10 bg-red-600 text-white px-3 py-1 rounded-md text-sm flex items-center gap-2">
                        <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                        Looping: {activeSegment.start}s - {activeSegment.end}s
                      </div>
                    )}
                    {activeVideoSrc ? (
                      <video
                        key={activeVideoSrc}
                        className="w-full h-full object-contain"
                        controls
                        autoPlay
                        src={activeVideoSrc}
                        ref={setVideoRef}
                      >
                        <source
                          src={activeVideoSrc}
                          type={`video/${getFilename(activeVideoPath)
                            .split(".")
                            .pop()}`}
                        />
                        Your browser does not support the video tag. (Path:{" "}
                        {activeVideoSrc})
                      </video>
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-gray-700">
                        <p>Video not available or not selected.</p>
                      </div>
                    )}
                  </div>

                  {analysisReport.data.find(
                    (item) => item.video_path === activeVideoPath
                  )?.feedback.final_conclusion && (
                    <div className="mt-4 p-4 bg-gray-800 rounded-md">
                      <h2 className="text-xl sm:text-2xl font-bold mb-2">
                        Summary for Current Segment:
                      </h2>
                      <p className="text-sm sm:text-base opacity-90">
                        {
                          analysisReport.data.find(
                            (item) => item.video_path === activeVideoPath
                          )?.feedback.final_conclusion
                        }
                      </p>
                    </div>
                  )}
                </div>

                <div
                  className="xl:col-span-1 rounded-lg p-0 sm:p-4"
                  style={{ backgroundColor: "#191919" }}
                >
                  <h2 className="text-xl sm:text-2xl font-bold mb-4 text-center sm:text-left">
                    Detailed Feedback Segments
                  </h2>

                  {/* Player Filter Dropdown */}
                  {availablePlayers.length > 0 && (
                    <div className="mb-4 p-3 bg-gray-800 rounded-lg">
                      <label className="block text-sm font-semibold mb-2">
                        Filter by Player:
                      </label>
                      <select
                        value={selectedPlayer ?? ""}
                        onChange={(e) =>
                          setSelectedPlayer(
                            e.target.value ? parseInt(e.target.value) : null
                          )
                        }
                        className="w-full bg-gray-700 text-white px-3 py-2 rounded text-sm focus:outline-none focus:ring-2 focus:ring-red-500"
                      >
                        <option value="">
                          All Players ({feedbackCount} total)
                        </option>
                        {availablePlayers.map((playerNum) => {
                          const playerCount =
                            analysisReport?.data?.filter(
                              (item) => item.player === playerNum
                            ).length || 0;
                          return (
                            <option key={playerNum} value={playerNum}>
                              Player #{playerNum} ({playerCount} feedback
                              {playerCount !== 1 ? "s" : ""})
                            </option>
                          );
                        })}
                      </select>
                    </div>
                  )}

                  <div className="max-h-[calc(100vh-200px)] overflow-y-auto pr-1 space-y-3">
                    {filteredData.map((item, index) => (
                      <CollapsibleFeedbackItem
                        key={`${index}-${item.video_path}`}
                        item={item}
                        index={index}
                        onSelectSegment={handleSelectSegment}
                        isActiveSegment={item.video_path === activeVideoPath}
                        onSeekToTimestamp={seekToTimestamp}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}

          {!error && (
            <div className="mt-12 relative">
              <button
                onClick={() => setSlideOut(true)}
                className="absolute left-1/2 top-1/2 transform -translate-x-1/2 translate-y-1/2 text-white hover:text-gray-400 transition-colors"
              >
                <ChevronDown className="h-8 w-8" />
              </button>
            </div>
          )}
        </main>
      </motion.div>
    </>
  );
}
