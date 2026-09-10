import type { Keypoint, Pose } from "../lib/poseTracking";

// A plausible standing BlazePose skeleton (33 points), normalized to the frame.
// Index order follows the BlazePose topology.
const DEFAULT: [number, number][] = [
  [0.5, 0.2], // 0 nose
  [0.48, 0.18], [0.47, 0.18], [0.46, 0.18], // 1-3 left eye
  [0.52, 0.18], [0.53, 0.18], [0.54, 0.18], // 4-6 right eye
  [0.44, 0.2], [0.56, 0.2], // 7-8 ears
  [0.48, 0.24], [0.52, 0.24], // 9-10 mouth
  [0.45, 0.35], [0.55, 0.35], // 11-12 shoulders
  [0.43, 0.5], [0.57, 0.5], // 13-14 elbows
  [0.42, 0.65], [0.58, 0.65], // 15-16 wrists
  [0.41, 0.69], [0.59, 0.69], // 17-18 pinky
  [0.42, 0.69], [0.58, 0.69], // 19-20 index
  [0.43, 0.67], [0.57, 0.67], // 21-22 thumb
  [0.47, 0.6], [0.53, 0.6], // 23-24 hips
  [0.47, 0.78], [0.53, 0.78], // 25-26 knees
  [0.47, 0.94], [0.53, 0.94], // 27-28 ankles
  [0.46, 0.96], [0.54, 0.96], // 29-30 heels
  [0.48, 0.98], [0.52, 0.98], // 31-32 foot index
];

/** Build a pose, optionally overriding specific landmark coordinates. */
export function makePose(
  overrides: Record<number, [number, number]> = {},
  visibility = 0.9
): Pose {
  return DEFAULT.map(([x, y], i): Keypoint => {
    const o = overrides[i];
    return { x: o ? o[0] : x, y: o ? o[1] : y, z: 0, visibility };
  });
}
