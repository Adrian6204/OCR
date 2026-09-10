// Minimal stand-in for @mediapipe/tasks-vision so lib modules import cleanly in
// Node. Only the named exports need to exist; no test calls into them.

export class FilesetResolver {
  static async forVisionTasks(): Promise<unknown> {
    return {};
  }
}

export class PoseLandmarker {
  static async createFromOptions(): Promise<PoseLandmarker> {
    return new PoseLandmarker();
  }
  detectForVideo(): unknown {
    return { landmarks: [] };
  }
}

export class ObjectDetector {
  static async createFromOptions(): Promise<ObjectDetector> {
    return new ObjectDetector();
  }
  detectForVideo(): unknown {
    return { detections: [] };
  }
}

export type PoseLandmarkerResult = { landmarks: unknown[] };
export type ObjectDetectorResult = { detections: unknown[] };
