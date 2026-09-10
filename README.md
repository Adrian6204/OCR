# Sentinel: CCTV-Style Hostile-Act Detection

A Next.js 14 + Tailwind proof-of-concept that watches a webcam feed like a CCTV camera, tracks everyone in frame, and flags likely **hostile acts** (fast strikes, lunges, close contact) with a live per-person threat score. Runs entirely on-device, no video leaves the machine, no backend.

> ⚠️ **Proof-of-concept, not a safety-critical system.** The threat score is a geometric heuristic, not a trained model. Expect false positives from high-fives, sports, or animated gestures. It's built to demonstrate the approach and its limits, not to be deployed.

## How it works

1. **Webcam**, `getUserMedia` streams the camera into a mirrored `<video>`.
2. **Multi-person pose**, MediaPipe `PoseLandmarker` (`@mediapipe/tasks-vision`) tracks up to 5 people, 33 body landmarks each, on-device in a `requestAnimationFrame` loop (`runningMode: "VIDEO"`). WASM + model load from a CDN.
3. **Tracking**, poses have no stable IDs across frames, so [threatDetect.ts](lib/threatDetect.ts) matches them frame-to-frame by torso-center proximity, giving each person a persistent id and velocity.
4. **Threat heuristic (Layer 1)**, per person, from geometry only:
   - **wrist speed** (fast motion),
   - **arm-extension rate** (a thrown punch extends the arm quickly),
   - **proximity** of a moving hand to another person's head/torso,
   - **stance** (both fists raised).
   A strike = fast + extending + closing on someone. Proximity gates the score so the same motion in empty space isn't flagged.
5. **Overlay + panel**, skeletons colored by level (teal → amber → red), bounding boxes, per-person threat readout, and a scene-level alert banner.

## Layer 2: trained model (optional, runs alongside the heuristic)

The heuristic is Layer 1. **Layer 2** adds a trained temporal classifier that
learns hostile-vs-normal motion over a ~1s window of pose keypoints, running
in-browser via onnxruntime-web *next to* the heuristic (shown as the "AI" verdict
in the panel). It is entirely optional: with no model file present, the app runs
on the heuristic alone.

The whole loop is self-contained, no external dataset required:

1. **Capture data**, open [`/capture`](app/capture/page.tsx), record labeled
   `normal` / `hostile` clips, and Export JSON. Capture exports *already-normalized*
   sequences, so [poseSequence.ts](lib/poseSequence.ts) is the single source of
   truth for feature extraction (no JS/Python drift).
2. **Train**, in [`ml/`](ml/): `pip install -r requirements.txt` then
   `python train.py --data ./data --epochs 80 --export` (PyTorch GRU, with
   train-time augmentation, early stopping, and precision/recall/F1 metrics).
3. **Deploy**, the export writes `public/violence.onnx`; reload the app and
   [classifier.ts](lib/classifier.ts) auto-detects and runs it (per-person,
   fused with the heuristic).

**Scale it up:** [`ml/preprocess_dataset.py`](ml/preprocess_dataset.py) runs
MediaPipe over a folder of labeled videos and emits the same training JSON, so
you can train on public datasets (RWF-2000, Hockey Fight) instead of only
hand-recorded clips. See [ml/README.md](ml/README.md).

## Testing

Pure logic (threat scoring, signal fusion, pose normalization) is covered by a
Vitest suite:

```bash
npm test
```

## Getting started

```bash
npm install
npm run dev
```

Open http://localhost:3000, allow camera access, and have one or two people move in frame. Camera access needs a secure context (`localhost` works; otherwise HTTPS).

## Project structure

```
app/
  page.tsx              -- monitor page: detection loop, incidents, AI signal
  capture/page.tsx      -- Layer 2 training-data collection tool
components/
  WebcamFeed.tsx        -- getUserMedia lifecycle + permission states
  PoseOverlay.tsx       -- multi-person skeleton canvas, colored by threat
  ThreatPanel.tsx       -- scene status + per-person breakdown + AI verdict
  CctvChrome.tsx        -- CCTV cosmetics (timestamp, REC, scanlines, vignette)
  IncidentLog.tsx       -- timestamped hostile-event snapshots
lib/
  poseTracking.ts       -- MediaPipe PoseLandmarker init + geometry helpers
  threatDetect.ts       -- person tracker + threat-score heuristics (Layer 1)
  poseSequence.ts       -- normalization + sequence buffer (data contract)
  classifier.ts         -- onnxruntime-web inference (Layer 2, model-gated)
ml/                     -- Python training project (PyTorch GRU → ONNX)
```

## Features

- **Weapon detection**, a MediaPipe ObjectDetector (COCO) flags held knives, scissors, and bats; a weapon near someone's hand marks that person **armed** and raises their threat (armed + aggressive motion → hostile). *Firearms are not a COCO class, so guns need a custom-trained detector, a Layer-2-style effort.*
- **Live threat scoring** with a scene banner + per-person breakdown and flags.
- **Incident log**, each hostile event is captured as a timestamped snapshot with its peak score.
- **Arm for unattended use**, one click unlocks an audio **alarm** and requests desktop **notifications**, so a hostile event actively alerts someone even with the tab in the background (browsers require that gesture to allow sound/notifications).
- **Set-once sensitivity**, persisted to the browser (localStorage) and reused automatically; no live babysitting.
- **CCTV chrome**, camera label, REC indicator, running clock, scanlines/vignette.

## Tuning

The heuristic thresholds live at the top of [threatDetect.ts](lib/threatDetect.ts) (`FAST_WRIST`, `EXT_RATE`, `PROXIMITY`, score bands), tune them against real footage to trade off sensitivity vs. false positives.
