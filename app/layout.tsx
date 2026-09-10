import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Sentinel · On-device Hostile-Act Detection",
  description:
    "A proof-of-concept CCTV-style safety monitor. On-device multi-person pose and weapon detection (MediaPipe) that reads body movement and flags likely hostile acts in real time. Runs entirely in the browser. Demo, not a production security system.",
  // A demonstration of a technique, not a deployable safety product. Keep it out
  // of search indexes so it is not mistaken for one.
  robots: { index: false, follow: false },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="app-bg min-h-screen font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
