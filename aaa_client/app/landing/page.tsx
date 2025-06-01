"use client";

import Link from "next/link";

export default function Component() {
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
        <Link href="/gallery" className="text-white text-lg">
          Login
        </Link>
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
