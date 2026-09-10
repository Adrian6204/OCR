"use client";

export interface Incident {
  id: string;
  /** Epoch ms when the incident was captured. */
  at: number;
  /** Peak threat score at trigger, 0..100. */
  peak: number;
  /** The person id that peaked, if known. */
  personId: number | null;
  /** Mirrored JPEG snapshot data URL. */
  thumb: string;
}

function formatTime(at: number): string {
  return new Date(at).toLocaleTimeString(undefined, {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/**
 * Timestamped log of hostile events, each with a snapshot, the "surveillance
 * record" that makes the monitor feel like a real product.
 */
export default function IncidentLog({
  incidents,
  onClear,
}: {
  incidents: Incident[];
  onClear: () => void;
}) {
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between border-b border-[var(--line)] px-5 py-3.5">
        <div className="flex items-center gap-2">
          <h2 className="text-[13px] font-semibold uppercase tracking-wider text-white/55">
            Incident log
          </h2>
          {incidents.length > 0 && (
            <span className="tnum rounded-full bg-[var(--hostile)]/15 px-2 py-0.5 text-[11px] font-semibold text-[var(--hostile)]">
              {incidents.length}
            </span>
          )}
        </div>
        {incidents.length > 0 && (
          <button
            onClick={onClear}
            className="rounded-md px-2 py-1 text-xs text-white/45 transition hover:bg-white/5 hover:text-white/80"
          >
            Clear
          </button>
        )}
      </div>

      {incidents.length === 0 ? (
        <p className="px-5 py-8 text-center text-sm text-white/40">
          No incidents yet. Hostile events are captured here with a timestamped
          snapshot.
        </p>
      ) : (
        <ul className="max-h-72 divide-y divide-[var(--line)] overflow-auto">
          {incidents.map((inc) => (
            <li
              key={inc.id}
              className="flex items-center gap-3 px-4 py-3 transition hover:bg-white/[0.02]"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={inc.thumb}
                alt={`Incident at ${formatTime(inc.at)}`}
                className="h-12 w-20 flex-shrink-0 rounded-md object-cover ring-1 ring-[var(--hostile)]/40"
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="tnum font-mono text-sm text-white/85">
                    {formatTime(inc.at)}
                  </span>
                  <span className="tnum rounded-full bg-[var(--hostile)]/15 px-2 py-0.5 text-[10px] font-semibold text-[var(--hostile)]">
                    THREAT {inc.peak}
                  </span>
                </div>
                <div className="mt-0.5 text-xs text-white/40">
                  {inc.personId !== null
                    ? `Person #${inc.personId}, hostile act`
                    : "Hostile act detected"}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
