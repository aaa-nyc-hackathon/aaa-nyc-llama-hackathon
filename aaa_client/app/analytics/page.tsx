"use client";

import { Search, ChevronDown } from "lucide-react";
import Link from "next/link";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function GameAnalysisPage() {
  const router = useRouter();
  const [slideOut, setSlideOut] = useState(false);

  return (
    <>
      <header className="flex items-center justify-between mb-8 bg-black text-white z-10 relative mx-auto px-6 sm:px-8 lg:px-12 py-4">
        <Link href="/landing" className="flex items-center gap-2">
          <img
            src="/logomark.png"
            alt="AthletIQ Logo"
            className="h-10 w-auto"
          />
          <span className="text-white text-xl font-bold">AthletIQ</span>
        </Link>

        <div className="relative max-w-xl w-full mx-4">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
          <input
            type="text"
            className="w-full rounded-full bg-gray-800/50 border border-gray-700 py-2 pl-10 pr-4 focus:outline-none focus:ring-1 focus:ring-gray-500"
            placeholder="Search"
          />
        </div>

        <div className="flex items-center gap-6">
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
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-4xl font-bold">Duke vs. Auburn 03-05-2025</h1>
            <span className="text-3xl font-semibold">
              5 Highlights, 3 Errors
            </span>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
            {/* Game Image and Summary (2/3 width) */}
            <div className="lg:col-span-2 space-y-4">
              <div className="relative aspect-video w-full bg-black rounded-md overflow-hidden">
                <video
                  className="w-full h-full object-cover"
                  controls
                  poster="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/Screenshot%202025-05-31%20at%2016.38.25.png-4AvOTasHRKvEtXY3kYj1qv6I5x4DaJ.jpeg"
                >
                  <source src="/placeholder.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>

              <div className="mt-4">
                <h2 className="text-2xl font-bold inline-block mr-2">
                  Game Summary:
                </h2>
                <span className="text-lg opacity-90">
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
                  do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                  Ut enim ad minim veniam, quis nostrud exercitation ullamco
                  laboris nisi ut aliquip ex ea commodo consequat.
                </span>
              </div>
            </div>

            {/* Highlights and Errors (1/3 width) */}
            <div
              className="lg:col-span-1 rounded-lg p-6"
              style={{ backgroundColor: "#191919" }}
            >
              <div className="mb-8">
                <h2 className="text-2xl font-bold mb-4">Highlights</h2>
                <ul className="space-y-4">
                  <li className="flex">
                    <span>1. Lorem ipsum dolor sit amet </span>
                    <span className="text-blue-400 ml-1">(03:51)</span>
                  </li>
                  <li className="flex">
                    <span>2. Lorem ipsum dolor sit amet </span>
                    <span className="text-blue-400 ml-1">(05:17)</span>
                  </li>
                  <li className="flex">
                    <span>3. Lorem ipsum dolor sit amet </span>
                    <span className="text-blue-400 ml-1">(07:34)</span>
                  </li>
                  <li className="flex">
                    <span>4. Lorem ipsum dolor sit amet </span>
                    <span className="text-blue-400 ml-1">(08:21)</span>
                  </li>
                  <li className="flex">
                    <span>5. Lorem ipsum dolor sit amet </span>
                    <span className="text-blue-400 ml-1">(10:09)</span>
                  </li>
                </ul>
              </div>

              <div>
                <h2 className="text-2xl font-bold mb-4">Errors</h2>
                <ul className="space-y-4">
                  <li className="flex">
                    <span>1. Lorem ipsum dolor sit amet </span>
                    <span className="text-blue-400 ml-1">(04:42)</span>
                  </li>
                  <li className="flex">
                    <span>2. Lorem ipsum dolor sit amet </span>
                    <span className="text-blue-400 ml-1">(06:23)</span>
                  </li>
                  <li className="flex">
                    <span>3. Lorem ipsum dolor sit amet </span>
                    <span className="text-blue-400 ml-1">(11:12)</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Close Button */}
          <div className="mt-12 relative">
            <button
              onClick={() => setSlideOut(true)}
              className="absolute left-1/2 top-1/2 transform -translate-x-1/2 translate-y-1/2 text-white"
            >
              <ChevronDown className="h-8 w-8" />
            </button>
          </div>
        </main>
      </motion.div>
    </>
  );
}
