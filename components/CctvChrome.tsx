"use client";

import { useEffect, useState } from "react";

/**
 * Purely cosmetic CCTV overlay: camera label, REC indicator, live timestamp,
 * scanlines and a vignette. Sits above the video, below the alert banner.
 */
export default function CctvChrome({ camId = "CAM 01" }: { camId?: string }) {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const stamp = now
    .toISOString()
    .replace("T", "  ")
    .replace(/\.\d+Z$/, " UTC");

  return (
    <div className="pointer-events-none absolute inset-0 z-10">
      {/* scanlines */}
      <div
        className="absolute inset-0 opacity-[0.07]"
        style={{
          backgroundImage:
            "repeating-linear-gradient(to bottom, #fff 0, #fff 1px, transparent 1px, transparent 3px)",
        }}
      />
      {/* vignette */}
      <div className="absolute inset-0 shadow-[inset_0_0_120px_rgba(0,0,0,0.65)]" />

      {/* top-left: cam id + REC */}
      <div className="absolute left-3 top-3 flex items-center gap-2 font-mono text-[11px] tracking-widest text-white/80">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
          REC
        </span>
        <span className="text-white/50">{camId}</span>
      </div>

      {/* bottom-right: timestamp */}
      <div className="absolute bottom-3 right-3 font-mono text-[11px] tracking-wider text-white/70">
        {stamp}
      </div>
    </div>
  );
}
