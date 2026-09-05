import { FEATURE_DIM, SEQ_LEN } from "./poseSequence";

// Optional Layer 2: run the trained ONNX classifier in-browser via
// onnxruntime-web. Everything here is lazy and model-gated — if
// /violence.onnx isn't present (no model trained yet), the classifier stays
// disabled and the app runs on the heuristic alone. onnxruntime-web is imported
// dynamically so it never enters the SSR/build graph.

const MODEL_URL = "/violence.onnx";
// Pinned to match the onnxruntime-web version in package.json.
const WASM_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

type OrtModule = typeof import("onnxruntime-web");

let ort: OrtModule | null = null;
let session: import("onnxruntime-web").InferenceSession | null = null;
let state: "idle" | "loading" | "ready" | "unavailable" = "idle";

/**
 * Attempt to load the classifier. Returns true if a model is ready, false if
 * none is available (or loading failed). Safe to call repeatedly.
 */
export async function loadClassifier(): Promise<boolean> {
  if (state === "ready") return true;
  if (state === "unavailable") return false;
  if (state === "loading") return false;
  state = "loading";

  try {
    // Cheap existence check before pulling in the (large) ORT runtime.
    const head = await fetch(MODEL_URL, { method: "HEAD" });
    if (!head.ok) {
      state = "unavailable";
      return false;
    }

    ort = await import("onnxruntime-web");
    ort.env.wasm.wasmPaths = WASM_CDN;
    session = await ort.InferenceSession.create(MODEL_URL);
    state = "ready";
    return true;
  } catch (err) {
    console.warn("Classifier unavailable:", err);
    state = "unavailable";
    return false;
  }
}

export function classifierReady(): boolean {
  return state === "ready";
}

function softmax2(a: number, b: number): number {
  const m = Math.max(a, b);
  const ea = Math.exp(a - m);
  const eb = Math.exp(b - m);
  return eb / (ea + eb); // P(class 1 = hostile)
}

/**
 * Run the model on one normalized sequence (length SEQ_LEN × FEATURE_DIM).
 * Returns P(hostile) in 0..1, or null if the model isn't ready.
 */
export async function classifySequence(
  seq: Float32Array
): Promise<number | null> {
  if (state !== "ready" || !ort || !session) return null;
  if (seq.length !== SEQ_LEN * FEATURE_DIM) return null;

  const input = new ort.Tensor("float32", seq, [1, SEQ_LEN, FEATURE_DIM]);
  const feeds: Record<string, import("onnxruntime-web").Tensor> = {
    [session.inputNames[0]]: input,
  };
  const output = await session.run(feeds);
  const logits = output[session.outputNames[0]].data as Float32Array;
  if (logits.length < 2) return null;
  return softmax2(logits[0], logits[1]);
}
