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
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Bell,
  Cpu,
  Crosshair,
  Play,
  ShieldEye,
  Square,
} from "@/components/icons";
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

  // Layer 2 classifier plumbing. Per-person sequence buffers (keyed by track id)
  // so several people can be scored and switching subjects doesn't corrupt a
  // sequence. Dormant unless a trained model is loaded.
  const buffersRef = useRef(new Map<number, SequenceBuffer>());
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
        /* audio unavailable, notifications may still work */
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

  // Load the object detector (optional, weapon detection degrades gracefully).
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

  // Detection loop, runs only once both the camera and model are ready.
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

          // Object detection is heavier, run it every 5th frame and reuse the
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

        // --- Layer 2: maintain a motion buffer per tracked person ---
        if (aiEnabledRef.current) {
          const buffers = buffersRef.current;
          const present = new Set(assessment.people.map((p) => p.id));
          for (const id of Array.from(buffers.keys())) {
            if (!present.has(id)) buffers.delete(id);
          }
          for (const p of assessment.people) {
            let b = buffers.get(p.id);
            if (!b) {
              b = new SequenceBuffer();
              buffers.set(p.id, b);
            }
            b.push(normalizePose(p.pose));
          }

          // Classify the highest-threat person whose buffer is ready (throttled,
          // non-overlapping). Its own buffer keeps history across subject switches.
          frameCounterRef.current++;
          const top = assessment.people.reduce<(typeof assessment.people)[number] | null>(
            (m, p) => (m && m.threat >= p.threat ? m : p),
            null
          );
          if (!top) setAiScore(null);
          if (frameCounterRef.current % 6 === 0 && !aiBusyRef.current && top) {
            const tensor = buffers.get(top.id)?.toFloat32();
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
      buffersRef.current.clear();
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
          /* notification failed, incident is still logged */
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fused.alert]);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-[var(--line)] bg-[var(--bg)]/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3 sm:px-6">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent/12 text-accent ring-1 ring-accent/25">
            <ShieldEye size={20} />
          </span>
          <div className="flex items-baseline gap-2.5">
            <span className="text-[15px] font-semibold tracking-tight text-white">
              Sentinel
            </span>
            <span className="hidden items-center gap-1.5 rounded-full border border-[var(--line)] bg-white/[0.03] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-white/45 sm:inline-flex">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
              Active development
            </span>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <StatusBadge
              modelStatus={modelStatus}
              people={scene.people.length}
              fps={fps}
            />
            <Link
              href="/capture"
              className="group hidden items-center gap-1.5 rounded-lg border border-[var(--line)] bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-white/60 transition hover:border-[var(--line-strong)] hover:text-white sm:inline-flex"
            >
              Capture data
              <ArrowRight
                size={13}
                className="transition-transform group-hover:translate-x-0.5"
              />
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 pb-16 sm:px-6">
        <section className="reveal reveal-1 pt-10 sm:pt-14">
          <h1 className="max-w-3xl text-3xl font-semibold leading-[1.1] tracking-tight text-white sm:text-[2.6rem]">
            Real-time hostile-act detection,{" "}
            <span className="text-accent">running entirely in your browser.</span>
          </h1>
          <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-white/55">
            Sentinel tracks everyone in frame, watches for weapons, and reads
            body movement for signs of a fight: fast strikes, lunges, and close
            contact. On-device MediaPipe means no video ever leaves your machine.
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            <Capability icon={<Cpu size={14} />} label="On-device, no server" />
            <Capability icon={<Crosshair size={14} />} label="Multi-person pose" />
            <Capability icon={<AlertTriangle size={14} />} label="Weapon-aware" />
            <Capability icon={<Activity size={14} />} label="Trainable model" />
          </div>
        </section>

        <div className="reveal reveal-2 mt-8 flex items-start gap-3 rounded-xl border border-amber-400/25 bg-amber-400/[0.07] px-4 py-3 text-sm text-amber-200/90">
          <AlertTriangle size={17} className="mt-0.5 shrink-0 text-amber-300" />
          <p className="leading-relaxed">
            <span className="font-semibold text-amber-200">
              Proof-of-concept demo, not a real security system.
            </span>{" "}
            Threat scores come from motion heuristics and can be wrong. A
            high-five, a fast reach, or a held kitchen knife may all trigger it.
            Please don&apos;t use it to make decisions about real people.
          </p>
        </div>

        <div className="reveal reveal-3 mt-6 grid gap-6 lg:grid-cols-[1.6fr_1fr]">
          <div className="space-y-5">
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
                  <div className="pointer-events-none absolute inset-0 z-30 rounded-2xl ring-[3px] ring-inset ring-[var(--hostile)]/70 animate-pulse" />
                  <div className="absolute right-3 top-3 z-30 flex items-center gap-2 rounded-full bg-[var(--hostile)] px-3 py-1.5 text-xs font-semibold text-white shadow-[0_8px_24px_-6px_rgba(251,106,104,0.6)]">
                    <span className="h-1.5 w-1.5 animate-ping rounded-full bg-white" />
                    Hostile activity
                  </div>
                </>
              )}
              {modelStatus === "loading" && cameraReady && (
                <div className="absolute bottom-3 left-3 z-30 rounded-full bg-black/70 px-3 py-1.5 text-xs text-white/70 backdrop-blur">
                  Loading pose model
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

            <IncidentLog incidents={incidents} onClear={() => setIncidents([])} />
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

      <SiteFooter />
    </div>
  );
}

function Capability({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-[var(--line)] bg-white/[0.02] px-3 py-1.5 text-xs font-medium text-white/60">
      <span className="text-accent">{icon}</span>
      {label}
    </span>
  );
}

function AboutSection() {
  return (
    <section className="mt-12 grid gap-10 border-t border-[var(--line)] pt-10 sm:grid-cols-2">
      <div>
        <h2 className="text-sm font-semibold text-white/85">How it works</h2>
        <ul className="mt-4 space-y-3 text-sm leading-relaxed text-white/50">
          <li>
            <span className="text-white/75">On-device only.</span> MediaPipe runs
            in your browser. No video ever leaves your machine, and there is no
            server.
          </li>
          <li>
            <span className="text-white/75">Pose and objects.</span> Multi-person
            body tracking plus an object detector that spots held knives,
            scissors, or bats.
          </li>
          <li>
            <span className="text-white/75">Threat heuristic.</span> A score
            built from wrist speed, arm extension, proximity to others, and
            whether someone is armed.
          </li>
          <li>
            <span className="text-white/75">Optional trained model.</span> A GRU
            can be trained on your own captured clips and fused with the
            heuristic (see the capture tool and the{" "}
            <code className="rounded bg-white/5 px-1 py-0.5 font-mono text-[12px] text-white/70">
              ml/
            </code>{" "}
            project).
          </li>
        </ul>
      </div>
      <div>
        <h2 className="text-sm font-semibold text-white/85">Limitations</h2>
        <ul className="mt-4 space-y-3 text-sm leading-relaxed text-white/50">
          <li>
            <span className="text-amber-300/85">False positives are expected.</span>{" "}
            The heuristic reacts to fast motion, not intent.
          </li>
          <li>
            <span className="text-amber-300/85">No firearm detection.</span> Guns
            aren&apos;t a COCO class, so only sharp or blunt held objects are
            caught.
          </li>
          <li>Single camera, single view. Occlusion and crowding reduce accuracy.</li>
          <li>
            Running two vision models is demanding, so expect lower framerates on
            low-end devices.
          </li>
          <li>
            This is a demonstration of a technique,{" "}
            <span className="text-white/75">not a validated safety product.</span>
          </li>
        </ul>
      </div>
    </section>
  );
}

function SiteFooter() {
  return (
    <footer className="border-t border-[var(--line)]">
      <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-8 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="flex items-start gap-2.5 text-xs text-white/45">
          <Activity size={15} className="mt-px shrink-0 text-accent" />
          <p className="max-w-md leading-relaxed">
            <span className="text-white/70">Continuously in development.</span>{" "}
            An evolving proof-of-concept. The heuristics and trained model keep
            improving as more data and tuning land.
          </p>
        </div>
        <div className="flex items-center gap-2 text-[11px] font-medium text-white/40">
          {["Next.js", "MediaPipe", "ONNX", "PyTorch"].map((t) => (
            <span
              key={t}
              className="rounded-md border border-[var(--line)] bg-white/[0.02] px-2 py-1"
            >
              {t}
            </span>
          ))}
        </div>
      </div>
    </footer>
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
    <div className="card space-y-3.5 p-5">
      <div className="flex flex-wrap items-center gap-3">
        <button
          onClick={onToggleArm}
          className={`inline-flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition ${
            armed
              ? "bg-[var(--hostile)]/12 text-[var(--hostile)] ring-1 ring-[var(--hostile)]/40 hover:bg-[var(--hostile)]/20"
              : "bg-accent text-[var(--accent-ink)] hover:brightness-110"
          }`}
        >
          {armed ? <Square size={13} /> : <Play size={13} />}
          {armed ? "Armed, monitoring" : "Arm monitor"}
        </button>

        <span className="flex items-center gap-1.5 text-xs text-white/45">
          <Bell size={13} className={armed ? "text-accent" : "text-white/30"} />
          {armed
            ? notifyOk
              ? "Alarm and desktop notifications active"
              : "Alarm active. Allow notifications for background alerts"
            : "Arm to enable the alarm and notifications, then leave it running"}
        </span>
      </div>

      <div className="border-t border-[var(--line)] pt-3.5">
        <label className="flex items-center gap-3">
          <span className="text-[11px] font-medium uppercase tracking-wider text-white/45">
            Sensitivity
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={sensitivity}
            onChange={(e) => onSensitivity(Number(e.target.value))}
            className="h-1.5 flex-1 cursor-pointer"
          />
          <span className="tnum w-10 text-right font-mono text-xs text-white/55">
            {Math.round(sensitivity * 100)}%
          </span>
        </label>
        <p className="mt-2 text-[11px] leading-relaxed text-white/35">
          Set once during setup. Saved to this browser and reused automatically.
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
      ? { color: "bg-[var(--hostile)]", label: "Model failed" }
      : modelStatus === "loading"
        ? { color: "bg-[var(--elevated)] animate-pulse", label: "Loading model" }
        : people > 0
          ? { color: "bg-accent", label: `${people} tracked · ${fps} fps` }
          : { color: "bg-white/40", label: "No one in frame" };

  return (
    <span className="tnum inline-flex items-center gap-2 rounded-full border border-[var(--line)] bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-white/70">
      <span className={`h-1.5 w-1.5 rounded-full ${color}`} />
      {label}
    </span>
  );
}
