"use client";

import type { Person, SceneAssessment, ThreatLevel } from "@/lib/threatDetect";
import { AGREEMENT_LABEL, type FusedAssessment } from "@/lib/fusion";
import { AlertTriangle, Crosshair } from "@/components/icons";

const LEVEL_META: Record<
  ThreatLevel,
  { label: string; text: string; bg: string; dot: string; bar: string }
> = {
  calm: {
    label: "Calm",
    text: "text-[var(--calm)]",
    bg: "bg-[var(--calm)]/[0.08]",
    dot: "bg-[var(--calm)]",
    bar: "bg-[var(--calm)]",
  },
  elevated: {
    label: "Elevated",
    text: "text-[var(--elevated)]",
    bg: "bg-[var(--elevated)]/[0.09]",
    dot: "bg-[var(--elevated)]",
    bar: "bg-[var(--elevated)]",
  },
  hostile: {
    label: "Hostile",
    text: "text-[var(--hostile)]",
    bg: "bg-[var(--hostile)]/[0.1]",
    dot: "bg-[var(--hostile)]",
    bar: "bg-[var(--hostile)]",
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
    <div className="card flex h-full flex-col gap-4 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-[13px] font-semibold uppercase tracking-wider text-white/55">
          Scene assessment
        </h2>
        <div className="flex items-center gap-2">
          {scene.weaponDetected && (
            <span className="inline-flex items-center gap-1 rounded-full bg-[var(--hostile)]/15 px-2.5 py-1 text-[11px] font-semibold text-[var(--hostile)]">
              <AlertTriangle size={12} />
              Weapon
            </span>
          )}
          <span className="tnum rounded-full border border-[var(--line)] bg-white/[0.03] px-2.5 py-1 text-[11px] font-medium text-white/50">
            {scene.people.length} tracked
          </span>
        </div>
      </div>

      {/* Fused status banner */}
      <div
        className={`rounded-xl p-4 ring-1 ring-inset ring-[var(--line)] ${meta.bg} ${
          fused.alert ? "animate-pulse" : ""
        }`}
      >
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${meta.dot}`} />
          <span className={`text-lg font-semibold ${meta.text}`}>
            {fused.alert ? "Hostile activity" : meta.label}
          </span>
        </div>
        <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-black/40">
          <div
            className={`h-full rounded-full ${meta.bar} transition-[width] duration-100`}
            style={{ width: `${fused.score}%` }}
          />
        </div>
        <div className="tnum mt-2 flex items-center justify-between font-mono text-xs text-white/45">
          <span>
            {fused.modelContributed ? "fused threat" : "threat"} {fused.score}/100
          </span>
          {fused.modelContributed && (
            <span className="text-white/35">{AGREEMENT_LABEL[fused.agreement]}</span>
          )}
        </div>
      </div>

      {/* Layer 2 model verdict, when a trained model is loaded */}
      {aiEnabled && (
        <div className="rounded-xl border border-[var(--line)] bg-black/25 p-3">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-2 text-xs font-medium text-white/60">
              <span className="rounded bg-accent/15 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-accent">
                AI
              </span>
              Trained model
            </span>
            <span
              className={`tnum font-mono text-sm font-semibold ${
                aiScore !== null && aiScore >= 0.6
                  ? "text-[var(--hostile)]"
                  : "text-white/70"
              }`}
            >
              {aiScore !== null ? `${Math.round(aiScore * 100)}%` : "..."}
            </span>
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-black/40">
            <div
              className={`h-full rounded-full transition-[width] duration-150 ${
                aiScore !== null && aiScore >= 0.6
                  ? "bg-[var(--hostile)]"
                  : "bg-accent"
              }`}
              style={{ width: `${Math.round((aiScore ?? 0) * 100)}%` }}
            />
          </div>
          <div className="mt-1.5 text-[10px] text-white/40">
            Probability of hostile motion over the last second
          </div>
        </div>
      )}

      {/* Per-person list */}
      <div className="flex-1 space-y-2 overflow-auto">
        {scene.people.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-2 pt-8 text-center text-white/40">
            <Crosshair size={22} className="text-white/25" />
            <p className="text-sm">No people detected. Step into frame.</p>
          </div>
        ) : (
          scene.people
            .slice()
            .sort((a, b) => b.threat - a.threat)
            .map((p) => <PersonRow key={p.id} person={p} />)
        )}
      </div>

      <p className="border-t border-[var(--line)] pt-3 text-[11px] leading-relaxed text-white/40">
        Threat is inferred from wrist speed, arm extension, and proximity to
        others. Expect false positives from high-fives, sports, or animated
        gestures.
      </p>
    </div>
  );
}

function PersonRow({ person }: { person: Person }) {
  const meta = LEVEL_META[person.level];
  return (
    <div className="rounded-lg border border-[var(--line)] bg-black/25 p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`h-2 w-2 rounded-full ${meta.dot}`} />
          <span className="tnum font-mono text-sm text-white/80">
            Person #{person.id}
          </span>
        </div>
        <span className={`tnum font-mono text-sm font-semibold ${meta.text}`}>
          {person.threat}
        </span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-black/40">
        <div
          className={`h-full rounded-full ${meta.bar} transition-[width] duration-100`}
          style={{ width: `${person.threat}%` }}
        />
      </div>
      {person.flags.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {person.flags.map((f) => (
            <span
              key={f}
              className="rounded-md border border-[var(--line)] bg-white/[0.03] px-2 py-0.5 text-[10px] text-white/60"
            >
              {f}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
