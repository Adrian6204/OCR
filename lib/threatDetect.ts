import {
  armExtension,
  dist,
  hasTorso,
  isVisible,
  L_ELBOW,
  L_SHOULDER,
  L_WRIST,
  poseQuality,
  poseScale,
  R_ELBOW,
  R_SHOULDER,
  R_WRIST,
  TARGET_POINTS,
  torsoCenter,
  type Keypoint,
  type Pose,
} from "./poseTracking";
import type { DetectedObject } from "./objectDetection";

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// Heuristic thresholds, in normalized-frame units (whole frame = 1.0).
// These are the knobs to tune against real footage.
const FAST_WRIST = 0.8; // units/sec that reads as a "fast" strike (peak-held)
const EXT_RATE = 1.5; // extension-ratio increase/sec that reads as a thrust
const PROXIMITY = 0.34; // wrist within this of another's body → "close contact"
const STALE_MS = 500; // drop a track unseen this long

// A strike is fast and brief, and often undersampled between frames, so we:
//  - peak-hold the combat signal (a spike survives a few frames), and
//  - use a fast-attack / slow-release envelope on the threat score so a punch
//    jumps to red instantly and lingers ~1s instead of being averaged away.
const COMBAT_DECAY = 0.6; // per-frame decay of the peak-held combat signal
const ATTACK = 0.6; // rise coefficient when the raw score exceeds current
const RELEASE = 0.06; // fall coefficient when it's below (slow decay)

// Accuracy gates that suppress "too much tracking":
const MIN_QUALITY = 0.5; // reject poses below this mean core-landmark visibility
const DEDUPE_FACTOR = 1.3; // merge poses whose centers are within this × body size
const MIN_HITS = 3; // frames a track must persist before it's shown (kills flicker)
const MATCH_FACTOR = 1.6; // match radius as a multiple of body size, clamped below
const MATCH_MIN = 0.06;
const MATCH_MAX = 0.32;

// Weapon association: an object is "held" if it's within this multiple of body
// size from a wrist (clamped). Being armed raises the score floor and, combined
// with aggressive motion, escalates to hostile.
const HOLD_FACTOR = 1.8;
const HOLD_MIN = 0.1;
const HOLD_MAX = 0.28;
const ARMED_FLOOR = 40; // armed alone → at least elevated
const ARMED_COMBAT_BOOST = 1.4; // armed AND moving → escalate

export type ThreatLevel = "calm" | "elevated" | "hostile";

export interface Person {
  id: number;
  pose: Pose;
  center: { x: number; y: number };
  /** Smoothed 0..100 threat score. */
  threat: number;
  level: ThreatLevel;
  /** Human-readable reasons contributing to the score. */
  flags: string[];
  /** A weapon/sharp object is associated with this person's hand. */
  armed: boolean;
  /** The weapon category, when armed. */
  weapon: string | null;
}

export interface SceneAssessment {
  people: Person[];
  /** Highest individual threat in the scene, 0..100. */
  maxThreat: number;
  level: ThreatLevel;
  /** True while any person is in the hostile band. */
  alert: boolean;
  /** True while any tracked person is armed. */
  weaponDetected: boolean;
}

interface Track {
  id: number;
  pose: Pose;
  center: { x: number; y: number };
  scale: number;
  /** Consecutive-ish frames this track has been matched — gates display. */
  hits: number;
  /** Peak-held combat signal (fast motion + thrust), decays each frame. */
  combatPeak: number;
  armed: boolean;
  weapon: string | null;
  prevLeftWrist: Keypoint | null;
  prevRightWrist: Keypoint | null;
  prevExtL: number | null;
  prevExtR: number | null;
  threat: number;
  flags: string[];
  lastSeen: number;
}

/** A cleaned, quality-filtered pose candidate for one frame. */
interface Candidate {
  pose: Pose;
  center: { x: number; y: number };
  scale: number;
  quality: number;
}

function levelFor(threat: number): ThreatLevel {
  if (threat >= 55) return "hostile";
  if (threat >= 25) return "elevated";
  return "calm";
}

/**
 * Stateful multi-person tracker + threat heuristic.
 *
 * MediaPipe returns poses without stable identities across frames, so we match
 * each frame's poses to existing tracks by torso-center proximity. Per track we
 * measure wrist speed, arm-extension rate, and closeness of a moving hand to
 * another person's body — the geometric signature of a strike. This is Layer 1:
 * fast, training-free, and honest about being heuristic (expect false positives
 * from high-fives, sports, or animated gestures).
 */
export class PersonTracker {
  private tracks: Track[] = [];
  private nextId = 1;
  private lastMs = -1;
  /** 0..1 — scales the threat score. 0.5 is neutral. */
  private sensitivity = 0.5;

  setSensitivity(value: number) {
    this.sensitivity = Math.min(1, Math.max(0, value));
  }

  update(
    poses: Pose[],
    objects: DetectedObject[],
    nowMs: number
  ): SceneAssessment {
    const dt = this.lastMs < 0 ? 1 / 30 : Math.min(0.2, (nowMs - this.lastMs) / 1000);
    this.lastMs = nowMs;

    const candidates = this.cleanPoses(poses);

    // --- Match candidates to existing tracks (greedy nearest-center, scaled) ---
    const usedTracks = new Set<number>();
    const assigned: (Track | null)[] = candidates.map(() => null);

    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      const radius = clamp(c.scale * MATCH_FACTOR, MATCH_MIN, MATCH_MAX);
      let best: Track | null = null;
      let bestD = radius;
      for (const t of this.tracks) {
        if (usedTracks.has(t.id)) continue;
        const d = Math.hypot(t.center.x - c.center.x, t.center.y - c.center.y);
        if (d < bestD) {
          bestD = d;
          best = t;
        }
      }
      if (best) {
        usedTracks.add(best.id);
        assigned[i] = best;
      }
    }

    // --- Update matched tracks / spawn new ones ---
    const liveTracks: Track[] = [];
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i];
      let track = assigned[i];
      if (!track) {
        track = {
          id: this.nextId++,
          pose: c.pose,
          center: c.center,
          scale: c.scale,
          hits: 0,
          combatPeak: 0,
          armed: false,
          weapon: null,
          prevLeftWrist: null,
          prevRightWrist: null,
          prevExtL: null,
          prevExtR: null,
          threat: 0,
          flags: [],
          lastSeen: nowMs,
        };
      }
      track.hits = Math.min(MIN_HITS + 3, track.hits + 1);
      track.scale = c.scale;
      this.updateTrack(track, c.pose, c.center, dt);
      liveTracks.push(track);
    }

    // Keep recently-seen unmatched tracks briefly (occlusion tolerance), but
    // decay their confidence so a flickering ghost dies instead of lingering.
    for (const t of this.tracks) {
      if (!liveTracks.includes(t) && nowMs - t.lastSeen < STALE_MS) {
        t.hits -= 1;
        if (t.hits > 0) liveTracks.push(t);
      }
    }
    this.tracks = liveTracks;

    // --- Weapon association: must run before scoring (armed affects threat) ---
    this.associateWeapons(objects);

    // --- Proximity + scoring pass: needs all people positioned this frame ---
    this.scoreProximity(dt);

    // Only surface confirmed tracks seen this frame — this is what removes the
    // swarm of single-frame phantom detections.
    const people: Person[] = this.tracks
      .filter((t) => t.lastSeen === nowMs && t.hits >= MIN_HITS)
      .map((t) => ({
        id: t.id,
        pose: t.pose,
        center: t.center,
        threat: Math.round(t.threat),
        level: levelFor(t.threat),
        flags: t.flags,
        armed: t.armed,
        weapon: t.weapon,
      }));

    const maxThreat = people.reduce((m, p) => Math.max(m, p.threat), 0);
    return {
      people,
      maxThreat,
      level: levelFor(maxThreat),
      alert: maxThreat >= 55,
      weaponDetected: people.some((p) => p.armed),
    };
  }

  /**
   * Link each detected weapon to the person whose nearest hand is closest,
   * within a body-scaled radius. Sets `armed`/`weapon` on the owning track and
   * adds a flag. A weapon with no hand nearby is left unassigned.
   */
  private associateWeapons(objects: DetectedObject[]) {
    const live = this.tracks.filter((t) => t.lastSeen === this.lastMs);
    for (const t of live) {
      t.armed = false;
      t.weapon = null;
    }
    if (objects.length === 0) return;

    for (const t of live) {
      const wrists = [t.pose[L_WRIST], t.pose[R_WRIST]].filter(isVisible);
      if (wrists.length === 0) continue;
      const radius = clamp(t.scale * HOLD_FACTOR, HOLD_MIN, HOLD_MAX);
      for (const obj of objects) {
        for (const w of wrists) {
          const d = Math.hypot(w.x - obj.center.x, w.y - obj.center.y);
          if (d <= radius) {
            t.armed = true;
            t.weapon = obj.category;
            t.flags.push(`armed: ${obj.category}`);
            break;
          }
        }
        if (t.armed) break;
      }
    }
  }

  /**
   * Reject junk poses and merge duplicates before tracking. This is the first
   * line of defense against over-tracking: drop partial/low-confidence
   * skeletons, then collapse overlapping detections of the same person.
   */
  private cleanPoses(poses: Pose[]): Candidate[] {
    const cands: Candidate[] = [];
    for (const pose of poses) {
      if (!hasTorso(pose)) continue;
      const center = torsoCenter(pose);
      if (!center) continue;
      const quality = poseQuality(pose);
      if (quality < MIN_QUALITY) continue;
      cands.push({ pose, center, scale: poseScale(pose), quality });
    }
    // Highest-quality first, then drop any later pose that overlaps a kept one.
    cands.sort((a, b) => b.quality - a.quality);
    const kept: Candidate[] = [];
    for (const c of cands) {
      const dupe = kept.some(
        (k) =>
          Math.hypot(k.center.x - c.center.x, k.center.y - c.center.y) <
          DEDUPE_FACTOR * Math.min(k.scale, c.scale)
      );
      if (!dupe) kept.push(c);
    }
    return kept;
  }

  private updateTrack(
    track: Track,
    pose: Pose,
    center: { x: number; y: number },
    dt: number
  ) {
    track.pose = pose;
    track.center = center;
    track.lastSeen = this.lastMs;

    // Wrist speeds (normalized units/sec).
    const lw = pose[L_WRIST];
    const rw = pose[R_WRIST];
    const lSpeed =
      isVisible(lw) && track.prevLeftWrist
        ? dist(lw, track.prevLeftWrist) / dt
        : 0;
    const rSpeed =
      isVisible(rw) && track.prevRightWrist
        ? dist(rw, track.prevRightWrist) / dt
        : 0;
    track.prevLeftWrist = isVisible(lw) ? lw : null;
    track.prevRightWrist = isVisible(rw) ? rw : null;

    // Arm extension rate (a thrust extends the arm quickly).
    const extL = armExtension(pose, L_SHOULDER, L_ELBOW, L_WRIST);
    const extR = armExtension(pose, R_SHOULDER, R_ELBOW, R_WRIST);
    const extRateL =
      extL !== null && track.prevExtL !== null
        ? Math.max(0, extL - track.prevExtL) / dt
        : 0;
    const extRateR =
      extR !== null && track.prevExtR !== null
        ? Math.max(0, extR - track.prevExtR) / dt
        : 0;
    track.prevExtL = extL;
    track.prevExtR = extR;

    const motion = clamp01(Math.max(lSpeed, rSpeed) / FAST_WRIST);
    const reach = clamp01(Math.max(extRateL, extRateR) / EXT_RATE);
    const combatInst = clamp01(0.6 * motion + 0.4 * reach);
    // Peak-hold so a punch that peaks between sampled frames still registers.
    track.combatPeak = Math.max(combatInst, track.combatPeak * COMBAT_DECAY);

    // Aggressive stance: both wrists raised above the shoulders.
    const raisedFists =
      isVisible(lw) &&
      isVisible(rw) &&
      isVisible(pose[L_SHOULDER]) &&
      isVisible(pose[R_SHOULDER]) &&
      lw.y < pose[L_SHOULDER].y &&
      rw.y < pose[R_SHOULDER].y;

    // Stash intermediate signals on the track for the proximity pass.
    track.flags = [];
    if (motion > 0.5) track.flags.push("fast motion");
    if (reach > 0.5) track.flags.push("striking motion");
    if (raisedFists) track.flags.push("raised fists");

    // Peak-held combat; proximity amplifies it in scoreProximity().
    (track as Track & { _combat?: number })._combat = track.combatPeak;
    (track as Track & { _stance?: number })._stance = raisedFists ? 25 : 0;
  }

  /** Amplify each person's score by how close a moving hand is to someone else. */
  private scoreProximity(_dt: number) {
    const live = this.tracks.filter((t) => t.lastSeen === this.lastMs);

    for (const t of live) {
      const wrists = [t.pose[L_WRIST], t.pose[R_WRIST]].filter(isVisible);
      let near = 0;
      for (const other of live) {
        if (other === t) continue;
        for (const idx of TARGET_POINTS) {
          const target = other.pose[idx];
          if (!isVisible(target)) continue;
          for (const w of wrists) {
            near = Math.max(near, clamp01((PROXIMITY - dist(w, target)) / PROXIMITY));
          }
        }
      }
      if (near > 0.5 && !t.flags.includes("close contact")) {
        t.flags.push("close contact");
      }

      const combat = (t as Track & { _combat?: number })._combat ?? 0;
      const stance = (t as Track & { _stance?: number })._stance ?? 0;
      // Proximity AMPLIFIES rather than gates: a strong fast strike can reach
      // the hostile band on its own (base 0.7), and closing on another person
      // pushes it well past it. This is what lets a real punch register even
      // when the fist doesn't land exactly on a tracked keypoint.
      let raw = 100 * combat * (0.7 + 0.5 * near);
      raw = Math.max(raw, stance);

      // Armed: hold a weapon → at least elevated; armed AND moving → escalate.
      if (t.armed) {
        raw = Math.max(raw, ARMED_FLOOR);
        if (combat > 0.4) raw = Math.min(100, raw * ARMED_COMBAT_BOOST);
      }

      // Sensitivity gain: 0.5 neutral, →1 amplifies, →0 attenuates.
      raw = Math.min(100, raw * (0.5 + this.sensitivity));

      // Fast-attack / slow-release: a brief strike spikes immediately, then
      // decays over ~1s so it's visible and can trigger, instead of being
      // averaged away by symmetric smoothing.
      const coeff = raw > t.threat ? ATTACK : RELEASE;
      t.threat = t.threat + coeff * (raw - t.threat);
    }
  }

  reset() {
    this.tracks = [];
    this.nextId = 1;
    this.lastMs = -1;
  }
}
