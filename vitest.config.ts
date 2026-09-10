import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

// Tests exercise the pure logic (fusion, threat scoring, normalization). The
// MediaPipe package is browser/WASM-only, so we alias it to a lightweight stub
//, none of the tested code paths actually call into it.
export default defineConfig({
  resolve: {
    alias: {
      "@mediapipe/tasks-vision": fileURLToPath(
        new URL("./test/stubs/tasks-vision.ts", import.meta.url)
      ),
      "@": fileURLToPath(new URL("./", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
  },
});
