import { describe, expect, it } from "vitest";
import { PersonTracker } from "../lib/threatDetect";
import { makePose } from "./helpers";
import type { DetectedObject } from "../lib/objectDetection";

const L_WRIST = 15;

/** Run the tracker over `frames` identical updates at 30fps. */
function run(
  poses: Parameters<PersonTracker["update"]>[0],
  objects: DetectedObject[],
  frames: number
) {
  const tracker = new PersonTracker();
  let scene = tracker.update(poses, objects, 0);
  for (let i = 1; i < frames; i++) {
    scene = tracker.update(poses, objects, i * 33);
  }
  return scene;
}

describe("PersonTracker", () => {
  it("reports calm with no one in frame", () => {
    const scene = new PersonTracker().update([], [], 0);
    expect(scene.people).toHaveLength(0);
    expect(scene.alert).toBe(false);
    expect(scene.level).toBe("calm");
  });

  it("confirms and tracks a stationary person as calm", () => {
    const scene = run([makePose()], [], 8);
    expect(scene.people).toHaveLength(1);
    expect(scene.people[0].level).toBe("calm");
    expect(scene.people[0].armed).toBe(false);
    expect(scene.weaponDetected).toBe(false);
  });

  it("does not show a person before the confirmation threshold", () => {
    const tracker = new PersonTracker();
    const scene = tracker.update([makePose()], [], 0); // first frame only
    expect(scene.people).toHaveLength(0);
  });

  it("marks a person armed and raises threat when holding a weapon", () => {
    const pose = makePose();
    const wrist = pose[L_WRIST];
    const knife: DetectedObject = {
      category: "knife",
      score: 0.9,
      box: { x: wrist.x - 0.02, y: wrist.y - 0.02, w: 0.04, h: 0.04 },
      center: { x: wrist.x, y: wrist.y },
    };
    const scene = run([pose], [knife], 10);
    expect(scene.people).toHaveLength(1);
    expect(scene.people[0].armed).toBe(true);
    expect(scene.people[0].weapon).toBe("knife");
    expect(scene.weaponDetected).toBe(true);
    // Armed floor lifts an otherwise-idle person to at least elevated.
    expect(scene.people[0].threat).toBeGreaterThanOrEqual(25);
    expect(scene.people[0].flags.some((f) => f.includes("knife"))).toBe(true);
  });
});
