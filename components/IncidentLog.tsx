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
 * Timestamped log of hostile events, each with a snapshot — the "surveillance
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
    <div className="rounded-2xl bg-panel ring-1 ring-white/10">
      <div className="flex items-center justify-between border-b border-white/10 px-5 py-3.5">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold tracking-wide text-white/80">
            Incident log
          </h2>
          {incidents.length > 0 && (
            <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-[11px] font-medium text-red-300">
              {incidents.length}
            </span>
          )}
        </div>
        {incidents.length > 0 && (
          <button
            onClick={onClear}
            className="text-xs text-white/40 transition hover:text-white/70"
          >
            Clear
          </button>
        )}
      </div>

      {incidents.length === 0 ? (
        <p className="px-5 py-8 text-center text-sm text-white/40">
          No incidents recorded. Hostile events are captured here with a
          timestamped snapshot.
        </p>
      ) : (
        <ul className="max-h-72 divide-y divide-white/5 overflow-auto">
          {incidents.map((inc) => (
            <li key={inc.id} className="flex items-center gap-3 px-4 py-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={inc.thumb}
                alt={`Incident at ${formatTime(inc.at)}`}
                className="h-12 w-20 flex-shrink-0 rounded object-cover ring-1 ring-red-500/40"
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm text-white/80 tabular-nums">
                    {formatTime(inc.at)}
                  </span>
                  <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-[10px] font-semibold text-red-300">
                    THREAT {inc.peak}
                  </span>
                </div>
                <div className="mt-0.5 text-xs text-white/40">
                  {inc.personId !== null
                    ? `Person #${inc.personId} · hostile act`
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
