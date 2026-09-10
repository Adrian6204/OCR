"""Turn a folder of labeled videos into training JSON — the scalable data path.

Instead of hand-recording clips in the web tool, point this at a public dataset
(RWF-2000, Hockey Fight, Movies Fight, …) or your own footage, and it runs
MediaPipe Pose over every clip, extracts normalized pose sequences, and writes
the SAME JSON format the trainer consumes.

Expected layout (label = subfolder name):

    videos/
      normal/   clip1.mp4 clip2.avi ...
      hostile/  fight1.mp4 fight2.mp4 ...

Usage:
    pip install -r requirements.txt mediapipe opencv-python
    # download the model once:
    #   curl -L -o pose_landmarker_full.task \\
    #     https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
    python preprocess_dataset.py --videos-dir ./videos --out ./data/dataset.json

CRITICAL: the normalization below is a line-for-line port of the web app's
`lib/poseSequence.ts` `normalizePose`. Both must stay identical, or preprocessed
data won't match what the model sees at inference. Keep SEQ_LEN / FEATURE_DIM in
sync with that file too.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

# These MUST match lib/poseSequence.ts.
SEQ_LEN = 32
NUM_KEYPOINTS = 33
FEATURE_DIM = NUM_KEYPOINTS * 2
FORMAT_VERSION = 1

# BlazePose indices (match lib/poseTracking.ts).
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
L_HIP, R_HIP = 23, 24
VISIBLE = 0.4

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _visible(lm) -> bool:
    return getattr(lm, "visibility", 0.0) >= VISIBLE


def normalize_pose(lms) -> list[float]:
    """Port of poseSequence.ts::normalizePose — keep in exact sync."""
    ls, rs = lms[L_SHOULDER], lms[R_SHOULDER]
    lh, rh = lms[L_HIP], lms[R_HIP]

    if _visible(lh) and _visible(rh):
        cx, cy = (lh.x + rh.x) / 2, (lh.y + rh.y) / 2
    elif _visible(ls) and _visible(rs):
        cx, cy = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
    else:
        cx = lms[NOSE].x if lms[NOSE] else 0.5
        cy = lms[NOSE].y if lms[NOSE] else 0.5

    scale = 0.0
    if _visible(ls) and _visible(rs) and _visible(lh) and _visible(rh):
        msx, msy = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2
        mhx, mhy = (lh.x + rh.x) / 2, (lh.y + rh.y) / 2
        scale = math.hypot(msx - mhx, msy - mhy)
    if scale < 1e-3 and _visible(ls) and _visible(rs):
        scale = math.hypot(ls.x - rs.x, ls.y - rs.y)
    if scale < 1e-3:
        scale = 0.2

    out: list[float] = []
    for i in range(NUM_KEYPOINTS):
        p = lms[i]
        out.append(((p.x if p else cx) - cx) / scale)
        out.append(((p.y if p else cy) - cy) / scale)
    return out


def _has_torso(lms) -> bool:
    return (
        _visible(lms[L_SHOULDER])
        and _visible(lms[R_SHOULDER])
        and (_visible(lms[L_HIP]) or _visible(lms[R_HIP]))
    )


def frames_from_video(path: str, landmarker) -> list[list[float]]:
    import cv2  # local import so the module loads without cv2 for --help
    import mediapipe as mp

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    seqs: list[list[float]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(idx * 1000.0 / fps)
        result = landmarker.detect_for_video(mp_image, ts)
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            if len(lms) >= NUM_KEYPOINTS and _has_torso(lms):
                seqs.append(normalize_pose(lms))
        idx += 1
    cap.release()
    return seqs


def windows(frames: list[list[float]], stride: int) -> list[list[list[float]]]:
    """Sliding SEQ_LEN windows → many samples per clip."""
    out = []
    for start in range(0, max(0, len(frames) - SEQ_LEN) + 1, stride):
        window = frames[start : start + SEQ_LEN]
        if len(window) == SEQ_LEN:
            out.append(window)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True,
                    help="dir with normal/ and hostile/ subfolders of videos")
    ap.add_argument("--out", default="./data/dataset.json")
    ap.add_argument("--model", default="pose_landmarker_full.task")
    ap.add_argument("--stride", type=int, default=SEQ_LEN // 2)
    args = ap.parse_args()

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Pose model not found at {args.model}. Download it first:\n"
            "  curl -L -o pose_landmarker_full.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task"
        )

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )

    samples = []
    for label in ("normal", "hostile"):
        folder = os.path.join(args.videos_dir, label)
        if not os.path.isdir(folder):
            print(f"⚠ Missing folder: {folder} — skipping.")
            continue
        paths = [
            p for p in sorted(glob.glob(os.path.join(folder, "*")))
            if p.lower().endswith(VIDEO_EXTS)
        ]
        print(f"[{label}] {len(paths)} videos in {folder}")
        # A fresh landmarker per label keeps the VIDEO-mode timestamp monotonic.
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            for path in paths:
                frames = frames_from_video(path, landmarker)
                wins = windows(frames, args.stride)
                for w in wins:
                    samples.append({"label": label, "seq": w})
                print(f"  {os.path.basename(path)}: {len(frames)} frames → {len(wins)} samples")

    if not samples:
        raise SystemExit("No samples produced — check the videos folder layout.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "version": FORMAT_VERSION,
        "seqLen": SEQ_LEN,
        "featureDim": FEATURE_DIM,
        "samples": samples,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    counts = {l: sum(1 for s in samples if s["label"] == l) for l in ("normal", "hostile")}
    print(f"Wrote {len(samples)} samples {counts} → {args.out}")


if __name__ == "__main__":
    main()
