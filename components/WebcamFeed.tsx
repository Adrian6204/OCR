"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Camera } from "@/components/icons";

export type CameraStatus =
  | "idle"
  | "requesting"
  | "ready"
  | "denied"
  | "error";

interface WebcamFeedProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  /** Fires once the stream is playing and has real dimensions. */
  onReady?: () => void;
  onStatusChange?: (status: CameraStatus) => void;
  /** Overlay layer(s) rendered above the video (e.g. the hand canvas). */
  children?: ReactNode;
}

/**
 * Owns the getUserMedia lifecycle and renders the mirrored <video> element.
 * Surfaces permission-denied and error states with a retry affordance.
 */
export default function WebcamFeed({
  videoRef,
  onReady,
  onStatusChange,
  children,
}: WebcamFeedProps) {
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const streamRef = useRef<MediaStream | null>(null);
  const [attempt, setAttempt] = useState(0);

  function update(next: CameraStatus) {
    setStatus(next);
    onStatusChange?.(next);
  }

  useEffect(() => {
    let cancelled = false;
    const video = videoRef.current;

    async function start() {
      update("requesting");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (video) {
          video.srcObject = stream;
          await video.play().catch(() => {
            /* autoplay may reject until metadata; onloadeddata retries */
          });
        }
      } catch (err) {
        if (cancelled) return;
        const e = err as DOMException;
        if (e.name === "NotAllowedError" || e.name === "SecurityError") {
          update("denied");
        } else {
          setErrorMsg(e.message || "Could not access the camera.");
          update("error");
        }
      }
    }

    start();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (video) video.srcObject = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt]);

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-2xl bg-black shadow-[var(--shadow-2)] ring-1 ring-[var(--line-strong)]">
      <video
        ref={videoRef}
        playsInline
        muted
        onLoadedData={() => {
          const v = videoRef.current;
          if (v && v.videoWidth > 0) {
            update("ready");
            onReady?.();
          }
        }}
        className="h-full w-full object-cover [transform:scaleX(-1)]"
      />

      {children}

      {status !== "ready" && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/75 backdrop-blur-sm">
          <div className="max-w-sm px-6 text-center">
            {status === "requesting" || status === "idle" ? (
              <>
                <Spinner />
                <p className="mt-4 text-sm text-white/70">
                  Requesting camera access
                </p>
              </>
            ) : status === "denied" ? (
              <>
                <span className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-[var(--elevated)]/12 text-[var(--elevated)]">
                  <Camera size={22} />
                </span>
                <p className="mt-4 text-lg font-medium text-white">
                  Camera access blocked
                </p>
                <p className="mt-2 text-sm text-white/60">
                  Allow camera permission for this site in your browser, then
                  retry.
                </p>
                <RetryButton onClick={() => setAttempt((a) => a + 1)} />
              </>
            ) : (
              <>
                <span className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-[var(--hostile)]/12 text-[var(--hostile)]">
                  <Camera size={22} />
                </span>
                <p className="mt-4 text-lg font-medium text-white">
                  Camera error
                </p>
                <p className="mt-2 text-sm text-white/60">{errorMsg}</p>
                <RetryButton onClick={() => setAttempt((a) => a + 1)} />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-white/15 border-t-accent" />
  );
}

function RetryButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="mt-5 rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-[var(--accent-ink)] transition hover:brightness-110"
    >
      Retry
    </button>
  );
}
