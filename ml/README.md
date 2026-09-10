# Layer 2 — Trained Violence Classifier

A temporal action-recognition model (PyTorch GRU) that learns to distinguish
**hostile** motion from **normal** motion over a ~1-second window of body-pose
keypoints. It runs *alongside* the geometric heuristic in the web app, exported
to ONNX and executed in-browser via onnxruntime-web.

## The data contract (why there's no normalization here)

The web capture tool (`/capture`) exports pose sequences that are **already
normalized** by the app's `lib/poseSequence.ts` (`normalizePose`): every frame
is centered on the mid-hip and scaled by torso length. That JS function is the
single source of truth, used for both the training data and live inference — so
this Python code never normalizes, it just consumes `[seq_len × feature_dim]`
vectors. No cross-language drift is possible.

- `seq_len = 32` frames (~1s at 30fps)
- `feature_dim = 66` (33 keypoints × x,y)
- labels: `normal` (0), `hostile` (1)

## Workflow

```
1. Collect data      →  open the web app at /capture, record clips
                        (aim for 30+ balanced samples per label), Export JSON
2. Install           →  cd ml && pip install -r requirements.txt
3. Train             →  python train.py --data ./data --epochs 60 --export
4. Run               →  the export writes ../public/violence.onnx;
                        reload the web app — it auto-detects and uses the model
```

Put the exported JSON files in `ml/data/` (or pass paths directly):

```bash
mkdir -p ml/data
# move your downloaded sentinel-samples-*.json into ml/data/
python train.py --data ./data --epochs 60 --export
```

`--export` writes ONNX to `../public/violence.onnx`. To export separately:

```bash
python export_onnx.py pose_gru.pt
```

## Files

| File | Role |
|---|---|
| `dataset.py` | Loads/validates JSON into tensors; train-time augmentation (flip/scale/jitter) via `SplitView` |
| `model.py` | `PoseGRU` — GRU + classification head |
| `train.py` | Stratified train/val/test split, augmentation, class weighting, early stopping, LR scheduling; saves best checkpoint + `metrics.json` |
| `export_onnx.py` | Checkpoint → `public/violence.onnx` (dynamic batch) |
| `preprocess_dataset.py` | Run MediaPipe over a folder of labeled videos → the same training JSON (the scalable data path) |

## Scaling up with public datasets

Hand-recording works for a first model, but the real accuracy lever is *more,
more varied* data. `preprocess_dataset.py` turns any folder of labeled videos
into training JSON — so you can train on public violence datasets (RWF-2000,
Hockey Fight, Movies Fight) or your own CCTV footage:

```
videos/
  normal/   clip1.mp4 ...
  hostile/  fight1.mp4 ...
```

```bash
pip install mediapipe opencv-python
curl -L -o pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
python preprocess_dataset.py --videos-dir ./videos --out ./data/dataset.json
python train.py --data ./data --epochs 80 --export
```

It slides a window over each clip to produce many samples, and its
normalization is a line-for-line port of the web app's `normalizePose`, so
preprocessed data is interchangeable with capture-tool data.

## Training rigor

`train.py` does a stratified **train/val/test** split, augments only the train
split, class-weights the loss for imbalance, uses **early stopping** + LR
scheduling, and writes **precision/recall/F1 + a confusion matrix** for the
hostile class to `metrics.json`. Use `--no-augment` to compare.

## Notes

- Start small: 30–50 samples per class already trains a usable POC. More data
  and more *variety* (distance, angle, speed, people) is the main accuracy
  lever.
- This is a proof-of-concept classifier, not a validated safety system.
- **Upgrade path:** to train on a public dataset (RWF-2000, Hockey Fight, …),
  write a preprocessing script that runs MediaPipe Pose over each clip and emits
  the same `{version, seqLen, featureDim, samples}` JSON — then `train.py` works
  unchanged. Swapping the GRU for an ST-GCN is a `model.py`-only change.
