type WindowWithWebkit = Window & { webkitAudioContext?: typeof AudioContext };

/**
 * Short audible alarm for hostile events. The AudioContext must be created from
 * a user gesture (browser autoplay policy), which is why the monitor is "armed"
 * with a click before it can sound, a natural one-time setup step for an
 * otherwise unattended system.
 */
export class Alarm {
  private ctx: AudioContext | null = null;

  get ready(): boolean {
    return !!this.ctx;
  }

  /** Create/resume the audio context. Call from a click/tap handler. */
  async unlock(): Promise<void> {
    if (this.ctx) {
      await this.ctx.resume();
      return;
    }
    const Ctor =
      window.AudioContext ?? (window as WindowWithWebkit).webkitAudioContext;
    if (!Ctor) throw new Error("Web Audio is not supported in this browser.");
    this.ctx = new Ctor();
    await this.ctx.resume();
  }

  /** Three quick square-wave pulses, a recognizable alert chirp. */
  beep(): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const start = ctx.currentTime;
    for (let i = 0; i < 3; i++) {
      const t = start + i * 0.18;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "square";
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(0, t);
      gain.gain.linearRampToValueAtTime(0.22, t + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.15);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t);
      osc.stop(t + 0.17);
    }
  }

  dispose(): void {
    this.ctx?.close().catch(() => {});
    this.ctx = null;
  }
}
