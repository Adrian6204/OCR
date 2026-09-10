import {
  isVisible,
  L_HIP,
  L_SHOULDER,
  NOSE,
  R_HIP,
  R_SHOULDER,
  type Pose,
} from "./poseTracking";

// ---------------------------------------------------------------------------
// The data contract shared by the capture tool (training-data producer) and the
// live classifier (inference consumer). Because BOTH normalize with the exact
// same code here, and the exported training JSON stores already-normalized
// vectors, the Python trainer never normalizes, it just consumes these
// vectors. That removes any chance of a JS/Python normalization mismatch.
// ---------------------------------------------------------------------------

/** Frames per classified sequence (~1s at 30fps). */
export const SEQ_LEN = 32;
export const NUM_KEYPOINTS = 33;
/** Features per frame: (x, y) per keypoint. */
export const FEATURE_DIM = NUM_KEYPOINTS * 2;

export const SEQUENCE_FORMAT_VERSION = 1;

/**
 * Normalize one pose into a translation- and scale-invariant feature vector:
 * center on the mid-hip (fallback mid-shoulder), scale by torso length. This is
 * what makes the model robust to where in the frame a person stands and how far
 * they are from the camera.
 */
export function normalizePose(pose: Pose): number[] {
  const ls = pose[L_SHOULDER];
  const rs = pose[R_SHOULDER];
  const lh = pose[L_HIP];
  const rh = pose[R_HIP];

  // Center: mid-hip, then mid-shoulder, then nose.
  let cx: number;
  let cy: number;
  if (isVisible(lh) && isVisible(rh)) {
    cx = (lh.x + rh.x) / 2;
    cy = (lh.y + rh.y) / 2;
  } else if (isVisible(ls) && isVisible(rs)) {
    cx = (ls.x + rs.x) / 2;
    cy = (ls.y + rs.y) / 2;
  } else {
    cx = pose[NOSE]?.x ?? 0.5;
    cy = pose[NOSE]?.y ?? 0.5;
  }

  // Scale: mid-shoulder→mid-hip torso length, else shoulder width, else default.
  let scale = 0;
  if (isVisible(ls) && isVisible(rs) && isVisible(lh) && isVisible(rh)) {
    const msx = (ls.x + rs.x) / 2;
    const msy = (ls.y + rs.y) / 2;
    const mhx = (lh.x + rh.x) / 2;
    const mhy = (lh.y + rh.y) / 2;
    scale = Math.hypot(msx - mhx, msy - mhy);
  }
  if (scale < 1e-3 && isVisible(ls) && isVisible(rs)) {
    scale = Math.hypot(ls.x - rs.x, ls.y - rs.y);
  }
  if (scale < 1e-3) scale = 0.2;

  const out: number[] = new Array(FEATURE_DIM);
  for (let i = 0; i < NUM_KEYPOINTS; i++) {
    const p = pose[i];
    out[i * 2] = ((p?.x ?? cx) - cx) / scale;
    out[i * 2 + 1] = ((p?.y ?? cy) - cy) / scale;
  }
  return out;
}

/**
 * Rolling buffer of the last SEQ_LEN normalized frames. Feed one normalized
 * frame per video frame; once `ready`, `toFloat32()` yields the flat
 * [SEQ_LEN × FEATURE_DIM] tensor the model expects.
 */
export class SequenceBuffer {
  private frames: number[][] = [];

  push(frame: number[]): void {
    this.frames.push(frame);
    if (this.frames.length > SEQ_LEN) this.frames.shift();
  }

  get ready(): boolean {
    return this.frames.length === SEQ_LEN;
  }

  clear(): void {
    this.frames = [];
  }

  toFloat32(): Float32Array | null {
    if (!this.ready) return null;
    const arr = new Float32Array(SEQ_LEN * FEATURE_DIM);
    for (let t = 0; t < SEQ_LEN; t++) {
      const frame = this.frames[t];
      for (let f = 0; f < FEATURE_DIM; f++) arr[t * FEATURE_DIM + f] = frame[f];
    }
    return arr;
  }
}

/** One labeled training sample: a normalized [SEQ_LEN × FEATURE_DIM] sequence. */
export interface LabeledSample {
  label: "normal" | "hostile";
  seq: number[][];
}

/** The exported training-data file shape consumed by the Python trainer. */
export interface SampleExport {
  version: number;
  seqLen: number;
  featureDim: number;
  samples: LabeledSample[];
}
