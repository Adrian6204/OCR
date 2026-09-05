"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import WebcamFeed from "@/components/WebcamFeed";
import PoseOverlay, { type PoseOverlayHandle } from "@/components/PoseOverlay";
import ThreatPanel from "@/components/ThreatPanel";
import CctvChrome from "@/components/CctvChrome";
import IncidentLog, { type Incident } from "@/components/IncidentLog";
import {
  detectPoses,
  getPoseLandmarker,
  type Pose,
  type PoseLandmarker,
} from "@/lib/poseTracking";
import {
  detectObjects,
  getObjectDetector,
  type DetectedObject,
  type ObjectDetector,
} from "@/lib/objectDetection";
import { PersonTracker, type SceneAssessment } from "@/lib/threatDetect";
import { normalizePose, SequenceBuffer } from "@/lib/poseSequence";
import { classifySequence, loadClassifier } from "@/lib/classifier";
import { fuse } from "@/lib/fusion";
import { Alarm } from "@/lib/alarm";
import Link from "next/link";

type ModelStatus = "loading" | "ready" | "error";

const IDLE_SCENE: SceneAssessment = {
  people: [],
  maxThreat: 0,
  level: "calm",
  alert: false,
  weaponDetected: false,
};

// Minimum gap between logged incidents so one sustained event isn't recorded
// dozens of times.
const INCIDENT_COOLDOWN_MS = 4000;
const MAX_INCIDENTS = 25;

/** Capture a small mirrored JPEG of the current video frame for the log. */
function captureThumb(video: HTMLVideoElement): string | null {
  if (video.videoWidth === 0) return null;
  const w = 160;
  const h = Math.round((w * video.videoHeight) / video.videoWidth) || 90;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.translate(w, 0);
  ctx.scale(-1, 1); // mirror to match the on-screen preview
  ctx.drawImage(video, 0, 0, w, h);
  return canvas.toDataURL("image/jpeg", 0.6);
}

export default function Page() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<PoseOverlayHandle>(null);

  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const objectDetectorRef = useRef<ObjectDetector | null>(null);
  const trackerRef = useRef(new PersonTracker());
  const rafRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const posesCacheRef = useRef<Pose[]>([]);
  const objectsCacheRef = useRef<DetectedObject[]>([]);
  const objFrameRef = useRef(0);
  const lastIncidentRef = useRef(0);

  // Layer 2 classifier plumbing.
  const seqBufferRef = useRef(new SequenceBuffer());
  const lastFedIdRef = useRef<number | null>(null);
  const aiEnabledRef = useRef(false);
  const aiBusyRef = useRef(false);
  const frameCounterRef = useRef(0);

  const [modelStatus, setModelStatus] = useState<ModelStatus>("loading");
  const [cameraReady, setCameraReady] = useState(false);
  const [scene, setScene] = useState<SceneAssessment>(IDLE_SCENE);
  const [fps, setFps] = useState(0);
  const [sensitivity, setSensitivity] = useState(0.5);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [aiEnabled, setAiEnabled] = useState(false);
  const [aiScore, setAiScore] = useState<number | null>(null);
  const [armed, setArmed] = useState(false);
  const [notifyOk, setNotifyOk] = useState(false);

  const alarmRef = useRef<Alarm | null>(null);
  const armedRef = useRef(false);

  // Load persisted sensitivity once (set-once config, survives reloads).
  useEffect(() => {
    const saved = localStorage.getItem("sentinel.sensitivity");
    if (saved !== null) setSensitivity(Number(saved));
  }, []);

  useEffect(() => {
    trackerRef.current.setSensitivity(sensitivity);
    localStorage.setItem("sentinel.sensitivity", String(sensitivity));
  }, [sensitivity]);

  useEffect(() => {
    return () => {
      alarmRef.current?.dispose();
      alarmRef.current = null;
    };
  }, []);

  async function toggleArm() {
    if (!armed) {
      if (!alarmRef.current) alarmRef.current = new Alarm();
      try {
        await alarmRef.current.unlock();
      } catch {
        /* audio unavailable — notifications may still work */
      }
      if (typeof Notification !== "undefined") {
        try {
          const perm =
            Notification.permission === "default"
              ? await Notification.requestPermission()
              : Notification.permission;
          setNotifyOk(perm === "granted");
        } catch {
          setNotifyOk(false);
        }
      }
      armedRef.current = true;
      setArmed(true);
    } else {
      armedRef.current = false;
      setArmed(false);
    }
  }

  // Try to load the optional trained classifier (no-op if no model is present).
  useEffect(() => {
    loadClassifier().then((ok) => {
      aiEnabledRef.current = ok;
      setAiEnabled(ok);
    });
  }, []);

  // Load the MediaPipe pose model once on mount.
  useEffect(() => {
    let active = true;
    getPoseLandmarker()
      .then((lm) => {
        if (!active) return;
        landmarkerRef.current = lm;
        setModelStatus("ready");
      })
      .catch((err) => {
        console.error("Failed to load PoseLandmarker:", err);
        if (active) setModelStatus("error");
      });
    return () => {
      active = false;
    };
  }, []);

  // Load the object detector (optional — weapon detection degrades gracefully).
  useEffect(() => {
    let active = true;
    getObjectDetector()
      .then((od) => {
        if (active) objectDetectorRef.current = od;
      })
      .catch((err) => console.warn("Object detector unavailable:", err));
    return () => {
      active = false;
    };
  }, []);

  // Detection loop — runs only once both the camera and model are ready.
  useEffect(() => {
    if (!cameraReady || modelStatus !== "ready") return;

    const tracker = trackerRef.current;
    tracker.reset();
    let lastFpsSample = performance.now();
    let frames = 0;

    const tick = () => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      if (video && landmarker && video.readyState >= 2) {
        let poses = posesCacheRef.current;
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          poses = detectPoses(landmarker, video, performance.now());
          posesCacheRef.current = poses;

          // Object detection is heavier — run it every 5th frame and reuse the
          // cached result in between (held objects move slowly relative to pose).
          const od = objectDetectorRef.current;
          if (od) {
            objFrameRef.current++;
            if (objFrameRef.current % 5 === 0) {
              try {
                objectsCacheRef.current = detectObjects(od, video, performance.now());
              } catch {
                /* skip this detection frame */
              }
            }
          }
        }

        const objects = objectsCacheRef.current;
        const assessment = tracker.update(poses, objects, performance.now());
        overlayRef.current?.draw(assessment.people, objects);
        setScene(assessment);

        // --- Layer 2: feed the highest-threat person's motion to the model ---
        if (aiEnabledRef.current) {
          const top = assessment.people.reduce<(typeof assessment.people)[number] | null>(
            (m, p) => (m && m.threat >= p.threat ? m : p),
            null
          );
          if (top) {
            // Reset the window when the tracked subject changes, so the
            // sequence stays coherent for one person.
            if (lastFedIdRef.current !== top.id) {
              seqBufferRef.current.clear();
              lastFedIdRef.current = top.id;
            }
            seqBufferRef.current.push(normalizePose(top.pose));
          } else {
            seqBufferRef.current.clear();
            lastFedIdRef.current = null;
            setAiScore(null);
          }

          frameCounterRef.current++;
          const buf = seqBufferRef.current;
          if (
            frameCounterRef.current % 6 === 0 &&
            buf.ready &&
            !aiBusyRef.current
          ) {
            const tensor = buf.toFloat32();
            if (tensor) {
              aiBusyRef.current = true;
              classifySequence(tensor)
                .then((p) => setAiScore(p))
                .finally(() => {
                  aiBusyRef.current = false;
                });
            }
          }
        }

        frames++;
        const now = performance.now();
        if (now - lastFpsSample >= 1000) {
          setFps(Math.round((frames * 1000) / (now - lastFpsSample)));
          frames = 0;
          lastFpsSample = now;
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastVideoTimeRef.current = -1;
      objectsCacheRef.current = [];
      overlayRef.current?.clear();
      setScene(IDLE_SCENE);
      seqBufferRef.current.clear();
      lastFedIdRef.current = null;
      setAiScore(null);
    };
  }, [cameraReady, modelStatus]);

  // Combine heuristic + model into one verdict (heuristic-only if no model).
  const fused = useMemo(
    () => fuse(scene, aiScore, aiEnabled),
    [scene, aiScore, aiEnabled]
  );

  // Record an incident on the rising edge of a fused hostile alert (debounced).
  useEffect(() => {
    if (!fused.alert) return;
    const now = Date.now();
    if (now - lastIncidentRef.current < INCIDENT_COOLDOWN_MS) return;
    const video = videoRef.current;
    const thumb = video ? captureThumb(video) : null;
    if (!thumb) return;
    lastIncidentRef.current = now;
    const top = [...scene.people].sort((a, b) => b.threat - a.threat)[0];
    setIncidents((prev) =>
      [
        {
          id:
            typeof crypto !== "undefined" && crypto.randomUUID
              ? crypto.randomUUID()
              : String(now),
          at: now,
          peak: fused.score,
          personId: top?.id ?? null,
          thumb,
        },
        ...prev,
      ].slice(0, MAX_INCIDENTS)
    );

    // Unattended alerting: sound the alarm and raise a desktop notification so
    // the event reaches someone even when nobody is watching the page.
    if (armedRef.current) {
      alarmRef.current?.beep();
      if (typeof Notification !== "undefined" && Notification.permission === "granted") {
        try {
          new Notification("⚠ Hostile activity detected", {
            body: `Threat ${fused.score}/100 · ${new Date(now).toLocaleTimeString()}`,
          });
        } catch {
          /* notification failed — incident is still logged */
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fused.alert]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:py-12">
      <div className="mb-6 flex items-start gap-3 rounded-xl border border-amber-400/25 bg-amber-400/10 px-4 py-3 text-sm text-amber-200/90">
        <span aria-hidden className="mt-0.5 text-base leading-none">⚠</span>
        <p>
          <span className="font-semibold">Proof-of-concept demo — not a real
          security system.</span>{" "}
          Threat scores come from motion heuristics and can be wrong (a
          high-five, a fast reach, or a held kitchen knife may all trigger).
          Don&apos;t use it to make decisions about real people.
        </p>
      </div>

      <header className="mb-8">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
            Sentinel
          </h1>
          <StatusBadge
            modelStatus={modelStatus}
            people={scene.people.length}
            fps={fps}
          />
          <Link
            href="/capture"
            className="ml-auto rounded-lg bg-white/5 px-3 py-1.5 text-xs font-medium text-white/60 ring-1 ring-white/10 transition hover:text-white"
          >
            Capture training data →
          </Link>
        </div>
        <p className="mt-2 max-w-2xl text-sm text-white/50">
          A CCTV-style safety monitor. On-device multi-person pose tracking
          (MediaPipe) reads body movement and flags likely hostile acts — fast
          strikes, lunges, close contact — with a live threat score. Runs
          entirely client-side; no video leaves your machine.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <div className="space-y-4">
          <WebcamFeed
            videoRef={videoRef}
            onReady={() => setCameraReady(true)}
            onStatusChange={(s) => {
              if (s !== "ready") setCameraReady(false);
            }}
          >
            <PoseOverlay
              ref={overlayRef}
              className="pointer-events-none absolute inset-0 z-20 h-full w-full"
            />
            <CctvChrome />
            {fused.alert && (
              <>
                <div className="pointer-events-none absolute inset-0 z-30 rounded-2xl ring-4 ring-inset ring-red-500/70 animate-pulse" />
                <div className="absolute right-3 top-3 z-30 flex items-center gap-2 rounded-full bg-red-500/90 px-3 py-1.5 text-xs font-semibold text-white shadow-lg">
                  <span className="h-2 w-2 animate-ping rounded-full bg-white" />
                  HOSTILE ACTIVITY
                </div>
              </>
            )}
            {modelStatus === "loading" && cameraReady && (
              <div className="absolute bottom-3 left-3 z-30 rounded-full bg-black/70 px-3 py-1.5 text-xs text-white/70 backdrop-blur">
                Loading pose model…
              </div>
            )}
          </WebcamFeed>

          <Controls
            armed={armed}
            notifyOk={notifyOk}
            onToggleArm={toggleArm}
            sensitivity={sensitivity}
            onSensitivity={setSensitivity}
          />

          <IncidentLog
            incidents={incidents}
            onClear={() => setIncidents([])}
          />
        </div>

        <div className="min-h-[420px]">
          <ThreatPanel
            scene={scene}
            fused={fused}
            aiEnabled={aiEnabled}
            aiScore={aiScore}
          />
        </div>
      </div>

      <AboutSection />
    </main>
  );
}

function AboutSection() {
  return (
    <section className="mt-10 border-t border-white/10 pt-8">
      <div className="grid gap-8 sm:grid-cols-2">
        <div>
          <h3 className="text-sm font-semibold text-white/80">How it works</h3>
          <ul className="mt-3 space-y-2 text-sm text-white/50">
            <li>
              <span className="text-white/70">On-device only.</span> MediaPipe
              runs in your browser — no video ever leaves your machine, and there
              is no server.
            </li>
            <li>
              <span className="text-white/70">Pose + objects.</span> Multi-person
              body tracking plus an object detector that spots held knives,
              scissors, or bats.
            </li>
            <li>
              <span className="text-white/70">Threat heuristic.</span> A score
              built from wrist speed, arm extension, proximity to others, and
              whether someone is armed.
            </li>
            <li>
              <span className="text-white/70">Optional trained model.</span> A
              GRU can be trained on your own captured clips and fused with the
              heuristic (see the capture tool + <code className="text-white/70">ml/</code>).
            </li>
          </ul>
        </div>
        <div>
          <h3 className="text-sm font-semibold text-white/80">Limitations</h3>
          <ul className="mt-3 space-y-2 text-sm text-white/50">
            <li>
              <span className="text-amber-300/80">False positives are
              expected</span> — the heuristic reacts to fast motion, not intent.
            </li>
            <li>
              <span className="text-amber-300/80">No firearm detection</span> —
              guns aren&apos;t a COCO class; only sharp/blunt held objects are
              caught.
            </li>
            <li>
              Single camera, single view; occlusion and crowding reduce accuracy.
            </li>
            <li>
              Running two vision models is demanding — expect lower framerates on
              low-end devices.
            </li>
            <li>
              This is a demonstration of a technique, <span className="text-white/70">not
              a validated safety product.</span>
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}

function Controls({
  armed,
  notifyOk,
  onToggleArm,
  sensitivity,
  onSensitivity,
}: {
  armed: boolean;
  notifyOk: boolean;
  onToggleArm: () => void;
  sensitivity: number;
  onSensitivity: (v: number) => void;
}) {
  return (
    <div className="space-y-3 rounded-2xl bg-panel px-5 py-4 ring-1 ring-white/10">
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={onToggleArm}
          className={`inline-flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition ${
            armed
              ? "bg-red-500/15 text-red-300 ring-1 ring-red-500/40 hover:bg-red-500/25"
              : "bg-accent text-ink hover:bg-accent/90"
          }`}
        >
          <span
            className={`h-2 w-2 rounded-full ${
              armed ? "animate-pulse bg-red-400" : "bg-ink/60"
            }`}
          />
          {armed ? "Armed — monitoring" : "Arm monitor"}
        </button>

        <span className="text-xs text-white/40">
          {armed
            ? notifyOk
              ? "Alarm + desktop notifications active"
              : "Alarm active · enable notifications for background alerts"
            : "Arm to enable the alarm and desktop notifications, then you can leave it running"}
        </span>
      </div>

      <div className="border-t border-white/5 pt-3">
        <label className="flex items-center gap-3">
          <span className="text-xs font-medium uppercase tracking-wide text-white/50">
            Sensitivity
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={sensitivity}
            onChange={(e) => onSensitivity(Number(e.target.value))}
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-white/10 accent-teal-400"
          />
          <span className="w-10 text-right font-mono text-xs text-white/50 tabular-nums">
            {Math.round(sensitivity * 100)}%
          </span>
        </label>
        <p className="mt-1.5 text-[11px] text-white/35">
          Set once during setup — saved to this browser and reused automatically.
        </p>
      </div>
    </div>
  );
}

function StatusBadge({
  modelStatus,
  people,
  fps,
}: {
  modelStatus: ModelStatus;
  people: number;
  fps: number;
}) {
  const { color, label } =
    modelStatus === "error"
      ? { color: "bg-red-400", label: "Model failed to load" }
      : modelStatus === "loading"
        ? { color: "bg-amber-400 animate-pulse", label: "Loading model" }
        : people > 0
          ? { color: "bg-accent", label: `${people} tracked · ${fps} fps` }
          : { color: "bg-white/40", label: "Monitoring · no one in frame" };

  return (
    <span className="inline-flex items-center gap-2 rounded-full bg-white/5 px-3 py-1 text-xs font-medium text-white/70 ring-1 ring-white/10">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}
