import { describe, expect, it } from "vitest";
import { fuse } from "../lib/fusion";
import type { SceneAssessment } from "../lib/threatDetect";

function scene(maxThreat: number): SceneAssessment {
  const level = maxThreat >= 55 ? "hostile" : maxThreat >= 25 ? "elevated" : "calm";
  return {
    people: [],
    maxThreat,
    level,
    alert: maxThreat >= 55,
    weaponDetected: false,
  };
}

describe("fuse", () => {
  it("returns the heuristic verdict when no model is enabled", () => {
    const f = fuse(scene(70), null, false);
    expect(f.modelContributed).toBe(false);
    expect(f.score).toBe(70);
    expect(f.alert).toBe(true);
    expect(f.agreement).toBe("none");
  });

  it("falls back to heuristic when enabled but score not ready", () => {
    const f = fuse(scene(30), null, true);
    expect(f.modelContributed).toBe(false);
    expect(f.score).toBe(30);
    expect(f.level).toBe("elevated");
  });

  it("blends heuristic and model when both present", () => {
    // h=0.4, m=0.9 → 100*(0.4*0.4 + 0.6*0.9) = 70
    const f = fuse(scene(40), 0.9, true);
    expect(f.modelContributed).toBe(true);
    expect(f.score).toBe(70);
    expect(f.alert).toBe(true);
    expect(f.agreement).toBe("model-only");
  });

  it("escalates with a corroboration boost when both agree", () => {
    // h=0.7, m=0.9 → base 100*(0.28+0.54)=82, ×1.15 boost = 94 (clamped 100)
    const f = fuse(scene(70), 0.9, true);
    expect(f.agreement).toBe("corroborated");
    expect(f.score).toBeGreaterThan(82);
    expect(f.alert).toBe(true);
  });

  it("stays quiet when neither signal is high", () => {
    const f = fuse(scene(10), 0.2, true);
    expect(f.agreement).toBe("quiet");
    expect(f.alert).toBe(false);
    expect(f.level).toBe("calm");
  });

  it("flags heuristic-only when motion is high but the model is unsure", () => {
    const f = fuse(scene(80), 0.2, true);
    expect(f.agreement).toBe("heuristic-only");
  });
});
