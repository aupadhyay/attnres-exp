import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";
import data from "./routing_data.json";

// Colors
const C_EMB = "#e85d26";   // orange
const C_NONE = "#c8ddf0";  // light blue

const KEYWORDS = new Set(["def", "if", "return", "for", "in", "else", "elif", "not", "and", "or", "pass", "while"]);
const BUILTINS = new Set(["len", "print", "range", "list", "int", "str"]);

function syntaxColor(tok: string): string {
  const s = tok.trim();
  if (KEYWORDS.has(s)) return "#3b5bdb";       // blue
  if (BUILTINS.has(s)) return "#7c3aed";       // purple
  if (/^\d+$/.test(s)) return "#2d6a4f";       // green
  if (/^[=<>\[\]():{},+\-*/#]+$/.test(s)) return "#555"; // operators
  return "#1a1a1a";
}

function lerpColor(c1: string, c2: string, t: number): string {
  const hex = (s: string) => [
    parseInt(s.slice(1,3),16),
    parseInt(s.slice(3,5),16),
    parseInt(s.slice(5,7),16)
  ];
  const [r1,g1,b1] = hex(c1);
  const [r2,g2,b2] = hex(c2);
  const r = Math.round(r1 + (r2-r1)*t);
  const g = Math.round(g1 + (g2-g1)*t);
  const b = Math.round(b1 + (b2-b1)*t);
  return `rgb(${r},${g},${b})`;
}

const MASK_NEWLINES = false; // toggle to suppress \n token dominance

export const CodeRouting: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const n = data.n_blocks;
  // Each block gets equal time, with smooth interpolation between them
  // Layout: [hold B1] [transition] [hold B2] [transition] ... [hold B4]
  const holdFrames = 30;      // 1s hold per block
  const finalHoldFrames = 105; // 3.5s hold on last block
  const transFrames = 15;     // 0.5s transition

  // Compute fractional block index (0 = B1, 3 = B4)
  // Total: (n-1) holds + (n-1) transitions + 1 final hold
  const totalFrames = (n - 1) * holdFrames + (n - 1) * transFrames + finalHoldFrames;
  const clampedFrame = Math.min(frame, totalFrames - 1);
  const t = Math.min(clampedFrame / (totalFrames - finalHoldFrames), 1) * (n - 1);
  const blockLo = Math.min(Math.floor(t), n - 2);
  const blockHi = blockLo + 1;
  const frac = t - blockLo;

  // Smooth frac with ease-in-out
  const smooth = frac < 0.5 ? 2*frac*frac : 1 - Math.pow(-2*frac+2,2)/2;

  // Interpolate emb values between adjacent blocks
  const embLo = data.emb_per_block[blockLo] as number[];
  const embHi = data.emb_per_block[blockHi] as number[];
  const emb = embLo.map((v, i) => v + (embHi[i] - v) * smooth);

  // Newline token indices (for optional masking)
  type TokenEntry = { tok: string; idx: number };
  const newlineIdxs = new Set(
    (data.lines as TokenEntry[][]).flatMap(line => line.filter(e => e.tok === "\n").map(e => e.idx))
  );

  const b0 = data.emb_per_block[0] as number[];
  const vmax = MASK_NEWLINES
    ? Math.max(...b0.filter((_, i) => !newlineIdxs.has(i)))
    : Math.max(...b0);

  // Which block label to show
  const blockLabel = t >= n - 1 ? n : Math.floor(t) + 1;

  // lines is now [{tok, idx}] — idx maps directly into emb_per_block
  const lines = data.lines as TokenEntry[][];

  return (
    <div style={{
      width: "100%", height: "100%",
      background: "white",
      fontFamily: "sans-serif",
      display: "flex",
      flexDirection: "column",
      padding: "40px 64px",
      boxSizing: "border-box",
    }}>
      {/* Title */}
      <div style={{ fontSize: 32, fontWeight: 700, marginBottom: 8, color: "#1a1a1a" }}>
        Depth attention routing — quicksort
      </div>
      <div style={{ fontSize: 22, color: "#666", marginBottom: 28 }}>
        Orange = attending to token embedding, Blue = attending to recent layers
      </div>

      {/* Block progress indicator */}
      <div style={{ display: "flex", gap: 20, marginBottom: 28, alignItems: "center" }}>
        {Array.from({length: n}, (_, i) => {
          const itemOpacity = Math.max(0.2, 1 - 0.78 * Math.max(0, t - i));
          const connReached = t > i;
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{
                width: 52, height: 52,
                borderRadius: "50%",
                background: t >= i ? C_EMB : "#ddd",
                opacity: itemOpacity,
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "white", fontWeight: 700, fontSize: 24,
              }}>
                {i+1}
              </div>
              {i < n-1 && <div style={{
                width: 56, height: 6,
                background: connReached ? C_EMB : "#ddd",
                opacity: itemOpacity,
                borderRadius: 3,
              }} />}
            </div>
          );
        })}
        <div style={{ marginLeft: 16, fontSize: 22, color: "#888" }}>
          Block boundary {blockLabel} of {n}
        </div>
      </div>

      {/* Code */}
      <div style={{
        fontFamily: "'Courier New', monospace",
        fontSize: 28,
        lineHeight: 1.6,
        background: "#f8f8f8",
        borderRadius: 12,
        padding: "24px 32px",
      }}>
        {lines.map((line, li) => (
          <div key={li} style={{ display: "flex", flexWrap: "nowrap", minHeight: "1.8em", alignItems: "stretch" }}>
            {line.map((entry, ti) => {
              const { tok, idx } = entry;
              const val = (MASK_NEWLINES && tok === "\n") ? 0 : (emb[idx] ?? 0);
              const normalized = Math.min(val / vmax, 1);
              const bg = lerpColor(C_NONE, C_EMB, normalized);
              const textColor = "white";
              if (tok === "\n") {
                // Render as a small colored end-of-line marker
                return (
                  <span key={`${li}-${ti}`} style={{
                    display: "block",
                    width: 28,
                    background: bg,
                    borderRadius: 0,
                    marginLeft: 2,
                    alignSelf: "stretch",
                  }} title="↵" />
                );
              }
              const isBold = KEYWORDS.has(tok.trim());
              return (
                <span key={`${li}-${ti}`} style={{
                  background: bg,
                  borderRadius: 0,
                  padding: "1px 0px",
                  whiteSpace: "pre",
                  color: textColor,
                  fontWeight: isBold ? 700 : 400,
                }}>
                  {entry.tok}
                </span>
              );
            })}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 20, marginTop: 28 }}>
        <span style={{ fontSize: 22, color: "#555", whiteSpace: "nowrap" }}>Recent layers</span>
        <div style={{
          flex: 1,
          height: 24,
          borderRadius: 8,
          background: `linear-gradient(to right, ${C_NONE}, ${C_EMB})`,
        }} />
        <span style={{ fontSize: 22, color: "#555", whiteSpace: "nowrap" }}>Token embedding</span>
      </div>
    </div>
  );
};
