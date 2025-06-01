"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function Component() {
  const [progress, setProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState("Initializing Analysis");
  const router = useRouter();

  const stages = [
    { text: "Initializing Analysis", minProgress: 0 },
    { text: "Processing Biometric Data", minProgress: 15 },
    { text: "Analyzing Performance Metrics", minProgress: 35 },
    { text: "Calculating Hydration Levels", minProgress: 55 },
    { text: "Refueling Electrolytes", minProgress: 75 },
    { text: "Notifying Talent Scouts", minProgress: 90 },
    { text: "Analysis Complete", minProgress: 100 },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        const newProgress = prev + Math.random() * 3 + 0.5;

        // Update stage based on progress
        const currentStageData = stages.find(
          (stage) =>
            newProgress >= stage.minProgress &&
            (stages[stages.indexOf(stage) + 1]?.minProgress > newProgress ||
              stage.minProgress === 100)
        );

        if (currentStageData && currentStageData.text !== currentStage) {
          setCurrentStage(currentStageData.text);
        }

        // Reset when complete
        if (newProgress >= 100) {
          setTimeout(() => {
            router.push("/analytics");
          }, 1000);
          return 100;
        }

        return newProgress;
      });
    }, 150);

    return () => clearInterval(interval);
  }, [currentStage, stages, router]);

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
          <label
            htmlFor="video-upload"
            className="cursor-pointer bg-gradient-to-r from-[#D9202C] to-[#731117] text-white px-6 py-2 rounded-full font-medium transition-all duration-300 ease-in-out hover:brightness-110 hover:scale-105 shadow-md hover:shadow-lg"
          >
            New Analysis
            <input
              id="video-upload"
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                if (e.target.files?.length) {
                  window.location.href = "/loading";
                }
              }}
            />
          </label>
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
        {/* Progress Bar */}
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
              {progress < 100 ? "Processing..." : "Complete"}
            </span>
          </div>
        </div>

        {/* Status Text */}
        <h2 className="text-2xl font-medium text-[#ffffff] text-center transition-all duration-500 ease-in-out">
          {currentStage}
        </h2>

        {/* Loading indicator dots */}
        {progress < 100 && (
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
      </main>
    </div>
  );
}
