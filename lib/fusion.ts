import type { SceneAssessment, ThreatLevel } from "./threatDetect";

// Fuse the two independent signals into a single verdict:
//   - Heuristic (Layer 1): instant and reactive, but false-positive prone.
//   - Trained model (Layer 2): steadier, better at "is this actually a fight",
//     but only present once a model has been trained.
// When no model is loaded, the fused verdict is exactly the heuristic. When
// both are present, the model is weighted higher, and agreement between the two
// escalates confidence (corroboration), the standard way to cut false alarms
// without going blind to real events.

const MODEL_WEIGHT = 0.6;
const HEURISTIC_WEIGHT = 0.4;
const CORROBORATION_BOOST = 1.15;

const H_HIGH = 0.6; // heuristic "confident" threshold (0..1)
const M_HIGH = 0.6; // model "confident" threshold (0..1)
const ALERT_AT = 62; // fused score that raises an alert (matches HOSTILE_AT)

export type Agreement =
  | "none" // no model contributing
  | "corroborated" // both signals high
  | "heuristic-only" // heuristic high, model not
  | "model-only" // model high, heuristic not
  | "quiet"; // neither high

export interface FusedAssessment {
  /** Combined 0..100 threat score. */
  score: number;
  level: ThreatLevel;
  alert: boolean;
  /** Whether the trained model contributed this frame. */
  modelContributed: boolean;
  agreement: Agreement;
}

function levelFor(score: number): ThreatLevel {
  if (score >= ALERT_AT) return "hostile";
  if (score >= 25) return "elevated";
  return "calm";
}

export function fuse(
  scene: SceneAssessment,
  aiScore: number | null,
  modelEnabled: boolean
): FusedAssessment {
  // No usable model signal → the heuristic is the verdict.
  if (!modelEnabled || aiScore === null) {
    return {
      score: scene.maxThreat,
      level: scene.level,
      alert: scene.alert,
      modelContributed: false,
      agreement: "none",
    };
  }

  const h = scene.maxThreat / 100;
  const m = aiScore;
  let score = 100 * (HEURISTIC_WEIGHT * h + MODEL_WEIGHT * m);

  const hHigh = h >= H_HIGH;
  const mHigh = m >= M_HIGH;
  if (hHigh && mHigh) score = Math.min(100, score * CORROBORATION_BOOST);

  const agreement: Agreement =
    hHigh && mHigh
      ? "corroborated"
      : hHigh
        ? "heuristic-only"
        : mHigh
          ? "model-only"
          : "quiet";

  const rounded = Math.round(score);
  return {
    score: rounded,
    level: levelFor(rounded),
    alert: rounded >= ALERT_AT,
    modelContributed: true,
    agreement,
  };
}

export const AGREEMENT_LABEL: Record<Agreement, string> = {
  none: "",
  corroborated: "Corroborated · heuristic + model agree",
  "heuristic-only": "Heuristic only · model unsure",
  "model-only": "Model only · motion subtle",
  quiet: "Both signals quiet",
};
