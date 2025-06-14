"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";

export default function Component() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [repeatPassword, setRepeatPassword] = useState("");
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center">
      {/* Main Content */}
      <div className="w-full max-w-sm space-y-6 px-6">
        {/* Email Field */}
        <div className="space-y-2">
          <Label htmlFor="email" className="text-white text-sm">
            Email
          </Label>
          <Input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="bg-black border-2 border-red-600 text-white placeholder:text-gray-400 focus:border-red-500 focus:ring-red-500 rounded-lg h-12"
            placeholder=""
          />
        </div>

        {/* Password Field */}
        <div className="space-y-2">
          <Label htmlFor="password" className="text-white text-sm">
            Password
          </Label>
          <Input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="bg-black border-2 border-red-600 text-white placeholder:text-gray-400 focus:border-red-500 focus:ring-red-500 rounded-lg h-12"
            placeholder=""
          />
        </div>

        {/* Repeat Password Field */}
        <div className="space-y-2">
          <Label htmlFor="repeat-password" className="text-white text-sm">
            Repeat Password
          </Label>
          <Input
            id="repeat-password"
            type="password"
            value={repeatPassword}
            onChange={(e) => setRepeatPassword(e.target.value)}
            className="bg-black border-2 border-red-600 text-white placeholder:text-gray-400 focus:border-red-500 focus:ring-red-500 rounded-lg h-12"
            placeholder=""
          />
        </div>

        {/* Terms Checkbox */}
        <div className="flex items-start space-x-2 pt-2">
          <Checkbox
            id="terms"
            checked={agreedToTerms}
            onCheckedChange={(checked) => setAgreedToTerms(checked as boolean)}
            className="border-white data-[state=checked]:bg-red-600 data-[state=checked]:border-red-600 mt-0.5"
          />
          <Label htmlFor="terms" className="text-white text-sm leading-relaxed">
            By continuing, you agree to our terms of service and privacy policy.
          </Label>
        </div>

        {/* Submit Button */}
        <Button className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold h-12 rounded-lg mt-6">
          {"Let's Go!"}
        </Button>

        {/* Divider */}
        <div className="flex items-center gap-4 py-4">
          <div className="flex-1 h-px bg-white"></div>
          <span className="text-white text-sm">OR</span>
          <div className="flex-1 h-px bg-white"></div>
        </div>

        {/* Social Login */}
        <div className="text-center">
          <p className="text-white text-sm mb-4">Register with</p>
          <div className="flex justify-center gap-6">
            {/* Google */}
            <button className="w-12 h-12 rounded-full bg-white flex items-center justify-center hover:bg-gray-100 transition-colors">
              <svg width="20" height="20" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                />
              </svg>
            </button>

            {/* Microsoft */}
            <button className="w-12 h-12 rounded-full bg-white flex items-center justify-center hover:bg-gray-100 transition-colors">
              <svg width="20" height="20" viewBox="0 0 24 24">
                <path fill="#F25022" d="M1 1h10v10H1z" />
                <path fill="#00A4EF" d="M13 1h10v10H13z" />
                <path fill="#7FBA00" d="M1 13h10v10H1z" />
                <path fill="#FFB900" d="M13 13h10v10H13z" />
              </svg>
            </button>

            {/* Apple */}
            <button className="w-12 h-12 rounded-full bg-white flex items-center justify-center hover:bg-gray-100 transition-colors">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="#000">
                <path d="M18.71 19.5c-.83 1.24-1.71 2.45-3.05 2.47-1.34.03-1.77-.79-3.29-.79-1.53 0-2 .77-3.27.82-1.31.05-2.3-1.32-3.14-2.53C4.25 17 2.94 12.45 4.7 9.39c.87-1.52 2.43-2.48 4.12-2.51 1.28-.02 2.5.87 3.29.87.78 0 2.26-1.07 3.81-.91.65.03 2.47.26 3.64 1.98-.09.06-2.17 1.28-2.15 3.81.03 3.02 2.65 4.03 2.68 4.04-.03.07-.42 1.44-1.38 2.83M13 3.5c.73-.83 1.94-1.46 2.94-1.5.13 1.17-.34 2.35-1.04 3.19-.69.85-1.83 1.51-2.95 1.42-.15-1.15.41-2.35 1.05-3.11z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
