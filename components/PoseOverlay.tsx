"use client";

import { forwardRef, useImperativeHandle, useRef } from "react";
import { isVisible } from "@/lib/poseTracking";
import type { Person, ThreatLevel } from "@/lib/threatDetect";
import type { DetectedObject } from "@/lib/objectDetection";

export interface PoseOverlayHandle {
  /** Draw tracked people (+ optional weapon boxes) for one frame. */
  draw: (people: Person[], objects?: DetectedObject[]) => void;
  clear: () => void;
}

const LEVEL_COLOR: Record<ThreatLevel, string> = {
  calm: "#5eead4", // teal
  elevated: "#fbbf24", // amber
  hostile: "#f87171", // red
};

/**
 * Transparent canvas overlay drawing every tracked person's skeleton, colored
 * by threat level, with a bounding box and threat readout. The canvas backing
 * store is sized to its rendered CSS size (not the video's intrinsic size) so
 * landmarks stay aligned, and x is mirrored to match the flipped preview.
 */
const PoseOverlay = forwardRef<PoseOverlayHandle, { className?: string }>(
  function PoseOverlay({ className }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    function syncSize(canvas: HTMLCanvasElement): { w: number; h: number } {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(rect.width * dpr);
      const h = Math.round(rect.height * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      return { w, h };
    }

    function drawPerson(
      ctx: CanvasRenderingContext2D,
      person: Person,
      w: number,
      h: number
    ) {
      const color = LEVEL_COLOR[person.level];

      // Bounding box around visible points (mirrored x). No skeleton, a clean
      // box keeps the view readable and puts the focus on the behavior label.
      const vis = person.pose.filter(isVisible);
      if (vis.length === 0) return;
      let minX = 1;
      let minY = 1;
      let maxX = 0;
      let maxY = 0;
      for (const p of vis) {
        minX = Math.min(minX, 1 - p.x);
        maxX = Math.max(maxX, 1 - p.x);
        minY = Math.min(minY, p.y);
        maxY = Math.max(maxY, p.y);
      }
      const pad = w * 0.012;
      const bx = minX * w;
      const by = minY * h;
      const bw = (maxX - minX) * w;
      const bh = (maxY - minY) * h;

      // Box, bolder as threat rises.
      ctx.strokeStyle = color;
      ctx.globalAlpha = person.level === "calm" ? 0.6 : 0.95;
      ctx.lineWidth = person.level === "hostile" ? 3.5 : 2;
      roundRect(ctx, bx - pad, by - pad, bw + pad * 2, bh + pad * 2, 10);
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Label chip above the box: id · level · score.
      const label = `#${person.id}  ${person.level.toUpperCase()}  ${person.threat}`;
      ctx.font = `600 ${Math.round(w * 0.02)}px ui-monospace, monospace`;
      const padX = w * 0.01;
      const textW = ctx.measureText(label).width;
      const chipH = w * 0.032;
      const chipX = bx - pad;
      const chipY = Math.max(0, by - pad - chipH - 3);
      ctx.fillStyle = color;
      roundRect(ctx, chipX, chipY, textW + padX * 2, chipH, 5);
      ctx.fill();
      ctx.fillStyle = "#0a0a0f";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, chipX + padX, chipY + chipH / 2);

      // Behavior tags under the box, the "what is this person doing" summary.
      if (person.flags.length > 0) {
        ctx.font = `500 ${Math.round(w * 0.017)}px ui-monospace, monospace`;
        ctx.fillStyle = color;
        ctx.textBaseline = "top";
        ctx.fillText(person.flags.join("  ·  "), chipX, by + bh + pad + 3);
      }
    }

    function drawObject(
      ctx: CanvasRenderingContext2D,
      obj: DetectedObject,
      w: number,
      h: number
    ) {
      // Mirror the box horizontally to match the flipped preview.
      const bx = (1 - obj.box.x - obj.box.w) * w;
      const by = obj.box.y * h;
      const bw = obj.box.w * w;
      const bh = obj.box.h * h;

      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#f87171";
      ctx.lineWidth = Math.max(2, w * 0.004);
      ctx.setLineDash([6, 4]);
      roundRect(ctx, bx, by, bw, bh, 6);
      ctx.stroke();
      ctx.setLineDash([]);

      const label = `⚠ ${obj.category} ${Math.round(obj.score * 100)}%`;
      ctx.font = `600 ${Math.round(w * 0.02)}px ui-monospace, monospace`;
      const padX = w * 0.008;
      const textW = ctx.measureText(label).width;
      const chipH = w * 0.028;
      ctx.fillStyle = "#f87171";
      roundRect(ctx, bx, Math.max(0, by - chipH - 3), textW + padX * 2, chipH, 4);
      ctx.fill();
      ctx.fillStyle = "#0a0a0f";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, bx + padX, Math.max(0, by - chipH - 3) + chipH / 2);
    }

    useImperativeHandle(ref, () => ({
      draw(people, objects = []) {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const { w, h } = syncSize(canvas);
        ctx.clearRect(0, 0, w, h);
        // Draw calm people first so hostile boxes render on top.
        const ordered = [...people].sort((a, b) => a.threat - b.threat);
        for (const person of ordered) drawPerson(ctx, person, w, h);
        for (const obj of objects) drawObject(ctx, obj, w, h);
        ctx.globalAlpha = 1;
      },
      clear() {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      },
    }));

    return <canvas ref={canvasRef} className={className} aria-hidden="true" />;
  }
);

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

export default PoseOverlay;
