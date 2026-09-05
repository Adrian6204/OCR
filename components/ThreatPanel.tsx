"use client";

import type { Person, SceneAssessment, ThreatLevel } from "@/lib/threatDetect";
import { AGREEMENT_LABEL, type FusedAssessment } from "@/lib/fusion";

const LEVEL_META: Record<
  ThreatLevel,
  { label: string; text: string; bg: string; dot: string; bar: string }
> = {
  calm: {
    label: "Calm",
    text: "text-teal-300",
    bg: "bg-teal-400/10",
    dot: "bg-teal-300",
    bar: "bg-teal-400",
  },
  elevated: {
    label: "Elevated",
    text: "text-amber-300",
    bg: "bg-amber-400/10",
    dot: "bg-amber-300",
    bar: "bg-amber-400",
  },
  hostile: {
    label: "Hostile",
    text: "text-red-300",
    bg: "bg-red-500/10",
    dot: "bg-red-400",
    bar: "bg-red-500",
  },
};

/**
 * Right-hand panel: scene-level threat status plus a per-person breakdown with
 * the heuristic flags that drove each score.
 */
export default function ThreatPanel({
  scene,
  fused,
  aiEnabled = false,
  aiScore = null,
}: {
  scene: SceneAssessment;
  fused: FusedAssessment;
  aiEnabled?: boolean;
  aiScore?: number | null;
}) {
  const meta = LEVEL_META[fused.level];

  return (
    <div className="flex h-full flex-col gap-4 rounded-2xl bg-panel p-5 ring-1 ring-white/10">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold tracking-wide text-white/80">
          Scene assessment
        </h2>
        <div className="flex items-center gap-2">
          {scene.weaponDetected && (
            <span className="rounded-full bg-red-500/15 px-2.5 py-1 text-[11px] font-semibold text-red-300">
              ⚠ Weapon
            </span>
          )}
          <span className="rounded-full bg-white/5 px-2.5 py-1 text-[11px] font-medium text-white/50">
            {scene.people.length} tracked
          </span>
        </div>
      </div>

      {/* Fused status banner */}
      <div
        className={`rounded-xl p-4 ring-1 ring-white/5 ${meta.bg} ${
          fused.alert ? "animate-pulse" : ""
        }`}
      >
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${meta.dot}`} />
          <span className={`text-lg font-semibold ${meta.text}`}>
            {fused.alert ? "⚠ Hostile activity" : meta.label}
          </span>
        </div>
        <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-white/10">
          <div
            className={`h-full rounded-full ${meta.bar} transition-[width] duration-100`}
            style={{ width: `${fused.score}%` }}
          />
        </div>
        <div className="mt-1.5 flex items-center justify-between font-mono text-xs text-white/40 tabular-nums">
          <span>
            {fused.modelContributed ? "fused threat" : "threat"} {fused.score}/100
          </span>
          {fused.modelContributed && (
            <span className="text-white/30">{AGREEMENT_LABEL[fused.agreement]}</span>
          )}
        </div>
      </div>

      {/* Layer 2 model verdict, when a trained model is loaded */}
      {aiEnabled && (
        <div className="rounded-xl bg-black/30 p-3 ring-1 ring-white/5">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-xs font-medium text-white/60">
              <span className="rounded bg-teal-400/15 px-1.5 py-0.5 text-[10px] font-semibold text-teal-300">
                AI
              </span>
              Trained model
            </span>
            <span
              className={`font-mono text-sm font-semibold tabular-nums ${
                aiScore !== null && aiScore >= 0.6
                  ? "text-red-300"
                  : "text-white/70"
              }`}
            >
              {aiScore !== null ? `${Math.round(aiScore * 100)}%` : "…"}
            </span>
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
            <div
              className={`h-full rounded-full transition-[width] duration-150 ${
                aiScore !== null && aiScore >= 0.6 ? "bg-red-500" : "bg-teal-400"
              }`}
              style={{ width: `${Math.round((aiScore ?? 0) * 100)}%` }}
            />
          </div>
          <div className="mt-1 text-[10px] text-white/40">
            P(hostile) over the last ~1s of motion
          </div>
        </div>
      )}

      {/* Per-person list */}
      <div className="flex-1 space-y-2 overflow-auto">
        {scene.people.length === 0 ? (
          <p className="pt-6 text-center text-sm text-white/40">
            No people detected. Step into frame.
          </p>
        ) : (
          scene.people
            .slice()
            .sort((a, b) => b.threat - a.threat)
            .map((p) => <PersonRow key={p.id} person={p} />)
        )}
      </div>

      <p className="text-[11px] leading-relaxed text-white/40">
        Heuristic proof-of-concept: threat is inferred from wrist speed, arm
        extension, and proximity to others — not a trained model. Expect false
        positives from high-fives, sports, or animated gestures.
      </p>
    </div>
  );
}

function PersonRow({ person }: { person: Person }) {
  const meta = LEVEL_META[person.level];
  return (
    <div className="rounded-lg bg-black/30 p-3 ring-1 ring-white/5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${meta.dot}`} />
          <span className="font-mono text-sm text-white/80">Person #{person.id}</span>
        </div>
        <span className={`font-mono text-sm font-semibold tabular-nums ${meta.text}`}>
          {person.threat}
        </span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className={`h-full rounded-full ${meta.bar} transition-[width] duration-100`}
          style={{ width: `${person.threat}%` }}
        />
      </div>
      {person.flags.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {person.flags.map((f) => (
            <span
              key={f}
              className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] text-white/60"
            >
              {f}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
