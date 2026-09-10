import { describe, expect, it } from "vitest";
import {
  FEATURE_DIM,
  SEQ_LEN,
  SequenceBuffer,
  normalizePose,
} from "../lib/poseSequence";
import { makePose } from "./helpers";
import type { Pose } from "../lib/poseTracking";

const MID_HIP = { x: 0.5, y: 0.6 };

function shift(pose: Pose, dx: number, dy: number): Pose {
  return pose.map((p) => ({ ...p, x: p.x + dx, y: p.y + dy }));
}

function scaleAbout(pose: Pose, k: number, cx: number, cy: number): Pose {
  return pose.map((p) => ({
    ...p,
    x: cx + (p.x - cx) * k,
    y: cy + (p.y - cy) * k,
  }));
}

describe("normalizePose", () => {
  it("produces a FEATURE_DIM-length vector", () => {
    expect(normalizePose(makePose())).toHaveLength(FEATURE_DIM);
  });

  it("is translation-invariant (centered on the hips)", () => {
    const base = normalizePose(makePose());
    const moved = normalizePose(shift(makePose(), 0.1, 0.05));
    for (let i = 0; i < base.length; i++) {
      expect(moved[i]).toBeCloseTo(base[i], 5);
    }
  });

  it("is scale-invariant (normalized by torso size)", () => {
    const base = normalizePose(makePose());
    const scaled = normalizePose(
      scaleAbout(makePose(), 1.35, MID_HIP.x, MID_HIP.y)
    );
    for (let i = 0; i < base.length; i++) {
      expect(scaled[i]).toBeCloseTo(base[i], 4);
    }
  });
});

describe("SequenceBuffer", () => {
  it("is not ready until SEQ_LEN frames are pushed", () => {
    const buf = new SequenceBuffer();
    const frame = normalizePose(makePose());
    for (let i = 0; i < SEQ_LEN - 1; i++) buf.push(frame);
    expect(buf.ready).toBe(false);
    expect(buf.toFloat32()).toBeNull();
    buf.push(frame);
    expect(buf.ready).toBe(true);
    expect(buf.toFloat32()).toHaveLength(SEQ_LEN * FEATURE_DIM);
  });

  it("clears back to empty", () => {
    const buf = new SequenceBuffer();
    const frame = normalizePose(makePose());
    for (let i = 0; i < SEQ_LEN; i++) buf.push(frame);
    buf.clear();
    expect(buf.ready).toBe(false);
  });
});
