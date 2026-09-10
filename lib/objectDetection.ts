import {
  FilesetResolver,
  ObjectDetector,
  type ObjectDetectorResult,
} from "@mediapipe/tasks-vision";

const WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
// EfficientDet-Lite0 (COCO, ~80 classes). Fast enough to run alongside pose at
// a throttled rate.
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.task";

// COCO categories we treat as weapons / sharp or blunt objects of concern.
// Note: COCO has no "gun"/"firearm" class, detecting firearms needs a custom
// trained model. These are the weapon-relevant classes COCO does provide.
export const WEAPON_CATEGORIES = ["knife", "scissors", "baseball bat"];

let detectorPromise: Promise<ObjectDetector> | null = null;

/** Lazily create a single ObjectDetector restricted to weapon categories. */
export function getObjectDetector(): Promise<ObjectDetector> {
  if (detectorPromise) return detectorPromise;

  detectorPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    return ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      scoreThreshold: 0.35,
      maxResults: 6,
      // Only surface weapon-relevant detections.
      categoryAllowlist: WEAPON_CATEGORIES,
    });
  })();

  return detectorPromise;
}

export type { ObjectDetector };

/** A detected object with a normalized (0..1, raw image) bounding box. */
export interface DetectedObject {
  category: string;
  score: number;
  box: { x: number; y: number; w: number; h: number };
  /** Box center, normalized, used to check which hand is holding it. */
  center: { x: number; y: number };
}

/**
 * Run object detection for one frame. `timestampMs` must be monotonically
 * increasing. Boxes are normalized against the video's intrinsic size so they
 * align with the pose landmarks (both in raw-image 0..1 space).
 */
export function detectObjects(
  detector: ObjectDetector,
  video: HTMLVideoElement,
  timestampMs: number
): DetectedObject[] {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const result: ObjectDetectorResult = detector.detectForVideo(
    video,
    timestampMs
  );

  const objects: DetectedObject[] = [];
  for (const det of result.detections) {
    const cat = det.categories[0];
    const bb = det.boundingBox;
    if (!cat || !bb) continue;
    const x = bb.originX / vw;
    const y = bb.originY / vh;
    const w = bb.width / vw;
    const h = bb.height / vh;
    objects.push({
      category: cat.categoryName || "object",
      score: cat.score,
      box: { x, y, w, h },
      center: { x: x + w / 2, y: y + h / 2 },
    });
  }
  return objects;
}
