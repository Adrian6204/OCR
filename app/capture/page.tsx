"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import WebcamFeed from "@/components/WebcamFeed";
import PoseOverlay, { type PoseOverlayHandle } from "@/components/PoseOverlay";
import {
  detectPoses,
  getPoseLandmarker,
  hasTorso,
  poseQuality,
  torsoCenter,
  type Pose,
  type PoseLandmarker,
} from "@/lib/poseTracking";
import {
  FEATURE_DIM,
  normalizePose,
  SEQ_LEN,
  SEQUENCE_FORMAT_VERSION,
  type LabeledSample,
  type SampleExport,
} from "@/lib/poseSequence";
import type { Person } from "@/lib/threatDetect";
import { ArrowLeft } from "@/components/icons";

type Label = "normal" | "hostile";
type ModelStatus = "loading" | "ready" | "error";

/** Highest-quality pose with a real torso, or null. */
function primaryPose(poses: Pose[]): Pose | null {
  let best: Pose | null = null;
  let bestQ = 0.4;
  for (const p of poses) {
    if (!hasTorso(p)) continue;
    const q = poseQuality(p);
    if (q > bestQ) {
      bestQ = q;
      best = p;
    }
  }
  return best;
}

export default function CapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<PoseOverlayHandle>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);

  // Recording state lives in refs so the rAF loop reads it without restarting.
  const recordingRef = useRef(false);
  const recordFramesRef = useRef<number[][]>([]);

  const [modelStatus, setModelStatus] = useState<ModelStatus>("loading");
  const [cameraReady, setCameraReady] = useState(false);
  const [label, setLabel] = useState<Label>("normal");
  const [recording, setRecording] = useState(false);
  const [progress, setProgress] = useState(0);
  const [samples, setSamples] = useState<LabeledSample[]>([]);

  useEffect(() => {
    let active = true;
    getPoseLandmarker()
      .then((lm) => {
        if (!active) return;
        landmarkerRef.current = lm;
        setModelStatus("ready");
      })
      .catch(() => active && setModelStatus("error"));
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!cameraReady || modelStatus !== "ready") return;

    const tick = () => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      if (video && landmarker && video.readyState >= 2) {
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          const poses = detectPoses(landmarker, video, performance.now());
          const primary = primaryPose(poses);

          if (primary) {
            const center = torsoCenter(primary) ?? { x: 0.5, y: 0.5 };
            const person: Person = {
              id: 0,
              pose: primary,
              center,
              threat: 0,
              level: "calm",
              flags: [],
              armed: false,
              weapon: null,
            };
            overlayRef.current?.draw([person]);

            if (recordingRef.current) {
              recordFramesRef.current.push(normalizePose(primary));
              const n = recordFramesRef.current.length;
              setProgress(n / SEQ_LEN);
              if (n >= SEQ_LEN) finishRecording();
            }
          } else {
            overlayRef.current?.draw([]);
            // Abort a recording if the person leaves frame.
            if (recordingRef.current) cancelRecording();
          }
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastVideoTimeRef.current = -1;
      overlayRef.current?.clear();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraReady, modelStatus]);

  function startRecording() {
    if (recordingRef.current || modelStatus !== "ready") return;
    recordFramesRef.current = [];
    recordingRef.current = true;
    setRecording(true);
    setProgress(0);
  }

  function finishRecording() {
    const frames = recordFramesRef.current.slice(0, SEQ_LEN);
    recordingRef.current = false;
    setRecording(false);
    setProgress(0);
    if (frames.length === SEQ_LEN) {
      setSamples((prev) => [...prev, { label, seq: frames }]);
    }
  }

  function cancelRecording() {
    recordingRef.current = false;
    recordFramesRef.current = [];
    setRecording(false);
    setProgress(0);
  }

  function exportJson() {
    const payload: SampleExport = {
      version: SEQUENCE_FORMAT_VERSION,
      seqLen: SEQ_LEN,
      featureDim: FEATURE_DIM,
      samples,
    };
    const blob = new Blob([JSON.stringify(payload)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sentinel-samples-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const counts = {
    normal: samples.filter((s) => s.label === "normal").length,
    hostile: samples.filter((s) => s.label === "hostile").length,
  };

  return (
    <main className="mx-auto max-w-4xl px-4 py-10 sm:px-6 lg:py-14">
      <header className="reveal reveal-1 mb-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-[1.9rem]">
            Training data capture
          </h1>
          <Link
            href="/"
            className="group inline-flex items-center gap-1.5 rounded-lg border border-[var(--line)] bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-white/60 transition hover:border-[var(--line-strong)] hover:text-white"
          >
            <ArrowLeft
              size={13}
              className="transition-transform group-hover:-translate-x-0.5"
            />
            Back to monitor
          </Link>
        </div>
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-white/55">
          Record {SEQ_LEN}-frame pose clips labeled{" "}
          <span className="text-[var(--calm)]">normal</span> or{" "}
          <span className="text-[var(--hostile)]">hostile</span>, then export
          JSON to train the Layer 2 model (see the{" "}
          <code className="rounded bg-white/5 px-1 py-0.5 font-mono text-[12px] text-white/70">
            ml/
          </code>{" "}
          project). Aim for 30 or more balanced samples per label, with variety
          in position, distance, and speed.
        </p>
      </header>

      <WebcamFeed
        videoRef={videoRef}
        onReady={() => setCameraReady(true)}
        onStatusChange={(s) => {
          if (s !== "ready") setCameraReady(false);
        }}
      >
        <PoseOverlay
          ref={overlayRef}
          className="pointer-events-none absolute inset-0 h-full w-full"
        />
        {recording && (
          <div className="absolute inset-x-0 bottom-0 z-40 h-1.5 bg-black/40">
            <div
              className="h-full bg-[var(--hostile)] transition-[width] duration-75"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>
        )}
        {recording && (
          <div className="absolute left-3 top-3 z-40 flex items-center gap-2 rounded-full bg-[var(--hostile)] px-3 py-1.5 text-xs font-semibold text-white">
            <span className="h-1.5 w-1.5 animate-ping rounded-full bg-white" />
            Recording
          </div>
        )}
      </WebcamFeed>

      <div className="reveal reveal-2 mt-5 flex flex-wrap items-center gap-3">
        <div className="inline-flex rounded-lg border border-[var(--line)] bg-white/[0.03] p-1">
          {(["normal", "hostile"] as Label[]).map((l) => (
            <button
              key={l}
              onClick={() => setLabel(l)}
              disabled={recording}
              className={`rounded-md px-4 py-2 text-sm font-medium capitalize transition disabled:opacity-50 ${
                label === l
                  ? l === "hostile"
                    ? "bg-[var(--hostile)] text-white"
                    : "bg-accent text-[var(--accent-ink)]"
                  : "text-white/55 hover:text-white"
              }`}
            >
              {l}
            </button>
          ))}
        </div>

        <button
          onClick={startRecording}
          disabled={recording || modelStatus !== "ready"}
          className="rounded-lg bg-accent px-5 py-2.5 text-sm font-semibold text-[var(--accent-ink)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:bg-white/10 disabled:text-white/40"
        >
          {recording ? "Recording" : `Record ${label} clip`}
        </button>

        <div className="tnum ml-auto flex items-center gap-4 text-sm">
          <span className="text-[var(--calm)]">normal {counts.normal}</span>
          <span className="text-[var(--hostile)]">hostile {counts.hostile}</span>
          <button
            onClick={exportJson}
            disabled={samples.length === 0}
            className="rounded-lg border border-[var(--line)] bg-white/[0.04] px-4 py-2 font-medium text-white transition hover:bg-white/[0.08] disabled:opacity-40"
          >
            Export JSON
          </button>
          <button
            onClick={() => setSamples([])}
            disabled={samples.length === 0}
            className="rounded-md px-2 py-1 text-white/45 transition hover:text-white/80 disabled:opacity-40"
          >
            Reset
          </button>
        </div>
      </div>

      <p className="reveal reveal-3 mt-4 text-xs leading-relaxed text-white/40">
        Each clip captures {SEQ_LEN} frames (about one second) of the
        highest-confidence person. Recording stops automatically when full, and
        cancels if the person leaves frame. Balanced classes matter, so record
        roughly equal counts.
      </p>
    </main>
  );
}
