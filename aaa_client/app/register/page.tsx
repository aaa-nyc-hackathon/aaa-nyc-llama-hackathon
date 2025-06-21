import Link from "next/link";
import Component from "@/components/ui/signup-form";

export default function Page() {
  return (
    <div className="min-h-screen bg-[#000000] text-white">
      {/* Navigation */}
      <nav className="flex items-center justify-between p-6">
        <Link href="/" className="flex items-center gap-2">
          <img
            src="/logomark.png"
            alt="AthletIQ Logo"
            className="h-10 w-auto"
          />
          <span className="text-white text-xl font-bold">AthletIQ</span>
        </Link>
        <div className="flex items-center gap-4">
          <Link href="/gallery" className="text-white text-lg">
            Login
          </Link>
        </div>
      </nav>
      <Component />
    </div>
  );
}