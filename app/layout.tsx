import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sentinel — CCTV Hostile-Act Detection (Demo)",
  description:
    "A proof-of-concept CCTV-style safety monitor. On-device multi-person pose tracking (MediaPipe) reads body movement and flags likely hostile acts with a live threat score. Demo only — not a real security system.",
  // This is a demonstration of a technique, not a deployable safety product —
  // keep it out of search indexes so it isn't mistaken for one.
  robots: { index: false, follow: false },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-ink bg-grid antialiased">
        {children}
      </body>
    </html>
  );
}
