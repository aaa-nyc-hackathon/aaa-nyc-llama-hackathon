"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChevronLeft, ChevronRight, Search, Menu } from "lucide-react";
import Image from "next/image";
import { useState } from "react";
import Link from "next/link";

export default function Component() {
  const [isSearchOpen, setIsSearchOpen] = useState(false);

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="flex items-center justify-between mb-8 bg-black text-white z-10 relative mx-auto px-6 sm:px-8 lg:px-12 py-4">
        <Link href="/landing" className="flex items-center gap-2">
          <img
            src="/logomark.png"
            alt="AthletIQ Logo"
            className="h-10 w-auto"
          />
          <span className="text-white text-xl font-bold">AthletIQ</span>
        </Link>

        {/* Desktop Search */}
        <div className="relative max-w-xl w-full mx-4">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            className="w-full rounded-full bg-gray-800/50 border border-gray-700 py-2 pl-10 pr-4 focus:outline-none focus:ring-1 focus:ring-gray-500"
            placeholder="Search"
          />
        </div>

        {/* Mobile Search Toggle */}
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

      {/* Mobile Search Bar */}
      {isSearchOpen && (
        <div className="md:hidden px-3 pb-3 bg-black">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              placeholder="Search..."
              className="w-full bg-gray-800 border-gray-700 text-white pl-10 rounded-full"
            />
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="relative">
        <div className="relative w-full h-[calc(100vh-60px)] sm:h-[calc(100vh-80px)] overflow-hidden">
          {/* Basketball Game Video/Image */}
          <Link href="/analytics" className="absolute inset-0">
            <Image
              src="/basketball-game.png"
              alt="Duke vs Auburn Basketball Game"
              fill
              className="object-cover w-auto h-full min-w-full"
              priority
            />
          </Link>

          {/* Navigation Arrows */}
          <Button
            variant="ghost"
            size="icon"
            className="absolute left-2 sm:left-4 top-1/2 transform -translate-y-1/2 w-8 h-8 sm:w-12 sm:h-12 rounded-full bg-black/20 hover:bg-black/40 text-white border border-white/20"
          >
            <ChevronLeft className="w-4 h-4 sm:w-6 sm:h-6" />
          </Button>

          <Button
            variant="ghost"
            size="icon"
            className="absolute right-2 sm:right-4 top-1/2 transform -translate-y-1/2 w-8 h-8 sm:w-12 sm:h-12 rounded-full bg-black/20 hover:bg-black/40 text-white border border-white/20"
          >
            <ChevronRight className="w-4 h-4 sm:w-6 sm:h-6" />
          </Button>

          {/* Game Information Overlay */}
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/100 to-transparent p-3 sm:p-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-end justify-between gap-2 sm:gap-0">
              <div>
                <h1 className="text-lg sm:text-2xl font-bold text-white">
                  <span className="block sm:hidden">Duke vs Auburn</span>
                  <span className="hidden sm:block">
                    Duke vs. Auburn 03-05-2025
                  </span>
                </h1>
                <p className="text-xs sm:hidden text-gray-300">03-05-2025</p>
              </div>
              <div>
                <p className="text-sm sm:text-xl font-semibold text-white">
                  <span className="block sm:hidden">
                    5 Highlights, 3 Errors
                  </span>
                  <span className="hidden sm:block">
                    5 Highlights, 3 Errors
                  </span>
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
