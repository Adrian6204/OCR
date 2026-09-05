import {
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";

// MediaPipe WASM + model assets are served from a CDN rather than bundled — the
// tasks-vision WASM files don't resolve cleanly through the normal Next import graph.
const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
// "full" is markedly more accurate than "lite" and produces far fewer phantom
// detections — worth the framerate cost for this use case. Swap to _heavy for
// the best accuracy, or _lite if the framerate is too low on weak hardware.
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";

/** Max simultaneous people to track (CCTV scenes are usually a handful). */
export const MAX_POSES = 4;

let landmarkerPromise: Promise<PoseLandmarker> | null = null;

/**
 * Lazily create a single multi-person PoseLandmarker configured for a video
 * stream. Cached across callers so the model is only fetched once.
 */
export function getPoseLandmarker(): Promise<PoseLandmarker> {
  if (landmarkerPromise) return landmarkerPromise;

  landmarkerPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: MAX_POSES,
      // Raised from 0.5 → 0.6 to suppress low-confidence phantom detections.
      minPoseDetectionConfidence: 0.6,
      minPosePresenceConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });
  })();

  return landmarkerPromise;
}

export type { PoseLandmarker, PoseLandmarkerResult };

/** One normalized body landmark (0..1 relative to the frame). */
export interface Keypoint {
  x: number;
  y: number;
  z: number;
  /** Model confidence this landmark is visible, 0..1. */
  visibility: number;
}

/** A single detected person: 33 BlazePose landmarks. */
export type Pose = Keypoint[];

/**
 * Run detection for one video frame. `timestampMs` must be monotonically
 * increasing across calls (VIDEO mode requirement).
 */
export function detectPoses(
  landmarker: PoseLandmarker,
  video: HTMLVideoElement,
  timestampMs: number
): Pose[] {
  const result: PoseLandmarkerResult = landmarker.detectForVideo(
    video,
    timestampMs
  );
  return result.landmarks.map((lm) =>
    lm.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z,
      visibility: p.visibility ?? 0,
    }))
  );
}

// BlazePose 33-point landmark indices (the ones we use).
export const NOSE = 0;
export const L_SHOULDER = 11;
export const R_SHOULDER = 12;
export const L_ELBOW = 13;
export const R_ELBOW = 14;
export const L_WRIST = 15;
export const R_WRIST = 16;
export const L_HIP = 23;
export const R_HIP = 24;
export const L_KNEE = 25;
export const R_KNEE = 26;
export const L_ANKLE = 27;
export const R_ANKLE = 28;

// Skeleton bones to draw (index pairs).
export const POSE_CONNECTIONS: ReadonlyArray<[number, number]> = [
  // Torso
  [L_SHOULDER, R_SHOULDER], [L_SHOULDER, L_HIP], [R_SHOULDER, R_HIP], [L_HIP, R_HIP],
  // Arms
  [L_SHOULDER, L_ELBOW], [L_ELBOW, L_WRIST],
  [R_SHOULDER, R_ELBOW], [R_ELBOW, R_WRIST],
  // Legs
  [L_HIP, L_KNEE], [L_KNEE, L_ANKLE],
  [R_HIP, R_KNEE], [R_KNEE, R_ANKLE],
];

/** Body parts a strike could target — used for proximity checks. */
export const TARGET_POINTS = [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP];

const VISIBLE = 0.4;

/** 2D distance between two keypoints (z ignored). */
export function dist(a: Keypoint, b: Keypoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Whether a keypoint is confidently visible. */
export function isVisible(p: Keypoint | undefined): p is Keypoint {
  return !!p && p.visibility >= VISIBLE;
}

/** Midpoint of two keypoints. */
export function mid(a: Keypoint, b: Keypoint): { x: number; y: number } {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/**
 * Torso-center of a pose (average of visible shoulders/hips), used to track a
 * person across frames. Falls back to the nose, then any visible point.
 */
export function torsoCenter(pose: Pose): { x: number; y: number } | null {
  const pts = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    .map((i) => pose[i])
    .filter(isVisible);
  if (pts.length > 0) {
    const x = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const y = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    return { x, y };
  }
  if (isVisible(pose[NOSE])) return { x: pose[NOSE].x, y: pose[NOSE].y };
  const any = pose.find(isVisible);
  return any ? { x: any.x, y: any.y } : null;
}

/**
 * A pose is "real" only if it has a confidently visible torso — both shoulders
 * and at least one hip. This rejects the partial/phantom skeletons the model
 * emits for background clutter or half-occluded regions.
 */
export function hasTorso(pose: Pose): boolean {
  return (
    isVisible(pose[L_SHOULDER]) &&
    isVisible(pose[R_SHOULDER]) &&
    (isVisible(pose[L_HIP]) || isVisible(pose[R_HIP]))
  );
}

/** Mean visibility over the core body landmarks, 0..1 — a quality score. */
export function poseQuality(pose: Pose): number {
  const core = [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE];
  let sum = 0;
  for (const i of core) sum += pose[i]?.visibility ?? 0;
  return sum / core.length;
}

/**
 * Approximate on-screen size of a person (shoulder width or torso height,
 * whichever is available), used to scale distance thresholds so matching works
 * for both near and far people. Falls back to a sane default.
 */
export function poseScale(pose: Pose): number {
  if (isVisible(pose[L_SHOULDER]) && isVisible(pose[R_SHOULDER])) {
    const shoulders = dist(pose[L_SHOULDER], pose[R_SHOULDER]);
    if (shoulders > 0.02) return shoulders;
  }
  const s = pose[L_SHOULDER] ?? pose[R_SHOULDER];
  const hip = pose[L_HIP] ?? pose[R_HIP];
  if (isVisible(s) && isVisible(hip)) return dist(s, hip);
  return 0.15;
}

/**
 * Arm extension ratio 0..1: how straight the arm is. ~1 = fully extended
 * (a thrown punch), lower = bent. Returns null if the arm isn't visible.
 */
export function armExtension(
  pose: Pose,
  shoulder: number,
  elbow: number,
  wrist: number
): number | null {
  const s = pose[shoulder];
  const e = pose[elbow];
  const w = pose[wrist];
  if (!isVisible(s) || !isVisible(e) || !isVisible(w)) return null;
  const upper = dist(s, e);
  const fore = dist(e, w);
  const span = dist(s, w);
  const limb = upper + fore;
  if (limb <= 1e-4) return null;
  return Math.min(1, span / limb);
}
