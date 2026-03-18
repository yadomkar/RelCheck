"""
RelCheck Architecture Diagram — Publication Quality v3
========================================================
Simplified top-down flow. Stage 2 uses a clean two-branch layout.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size": 9,
    "text.usetex": False,
    "figure.dpi": 300,
})

# Colors
BG       = "white"
C_IN     = "#DCEEFB"
C_MODEL  = "#BBDEFB"
C_S1     = "#FFF3E0"
C_SPAT   = "#C8E6C9"
C_VQA    = "#E1BEE7"
C_S3     = "#FFCCBC"
C_OUT    = "#B2DFDB"
C_DECIDE = "#FFF9C4"
C_EDGE   = "#37474F"
C_ARR    = "#455A64"
C_TXT    = "#212121"
C_GRAY   = "#9E9E9E"

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)

# ═══════════════════════ Helpers ═══════════════════════════════════════════

def rbox(x, y, w, h, txt, fc, fs=8.5, bold=False, ec=C_EDGE, lw=0.8, tc=C_TXT, rad=0.12):
    p = FancyBboxPatch((x,y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    wt = "bold" if bold else "normal"
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center",
            fontsize=fs, fontweight=wt, color=tc, zorder=3, linespacing=1.35)

def arr(x1, y1, x2, y2, c=C_ARR, lw=1.0, sty="-|>", ls="-", ms=11):
    a = FancyArrowPatch((x1,y1),(x2,y2), arrowstyle=sty, color=c,
        linewidth=lw, mutation_scale=ms, zorder=4, linestyle=ls)
    ax.add_patch(a)

def dia(cx, cy, w, h, txt, fc=C_DECIDE, fs=7.5):
    hw, hh = w/2, h/2
    v = [(cx,cy+hh),(cx+hw,cy),(cx,cy-hh),(cx-hw,cy),(cx,cy+hh)]
    ax.add_patch(plt.Polygon(v, fc=fc, ec=C_EDGE, lw=0.8, zorder=2))
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=C_TXT, zorder=3, linespacing=1.3)

def note(x, y, txt, fs=6.5, c=C_GRAY, style="italic", ha="center"):
    ax.text(x, y, txt, ha=ha, va="center", fontsize=fs, color=c,
            fontstyle=style, zorder=5)

def stag(x, y, txt, bg, fg, fs=6):
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=fg, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc=bg, ec=fg, lw=0.4))


# ═══════════════════════ TITLE ════════════════════════════════════════════
ax.text(7, 8.6,
    "RelCheck: Training-Free Detection and Correction of Relational Hallucinations",
    ha="center", va="center", fontsize=12, fontweight="bold", color=C_TXT)

# ═══════════════════════ STAGE LABELS (row at top) ════════════════════════
ax.text(1.5, 8.05, "Input", ha="center", fontsize=10, fontweight="bold",
        color="#546E7A", fontstyle="italic")
ax.text(4.2, 8.05, "Stage 1: Extract", ha="center", fontsize=10,
        fontweight="bold", color="#546E7A", fontstyle="italic")
ax.text(8.2, 8.05, "Stage 2: Verify", ha="center", fontsize=10,
        fontweight="bold", color="#546E7A", fontstyle="italic")
ax.text(12.0, 8.05, "Stage 3: Correct", ha="center", fontsize=10,
        fontweight="bold", color="#546E7A", fontstyle="italic")

# ═══════════ Stage dividers ════════════════════════════════════════════════
for xp in [2.8, 5.8, 10.1]:
    ax.plot([xp, xp], [0.6, 7.7], color="#E0E0E0", lw=0.5,
            ls=(0,(5,5)), zorder=0)


# ══════════════════════ INPUT (col 1) ═════════════════════════════════════
rbox(0.6, 6.4, 1.8, 0.9, "Input\nImage", C_IN, fs=9, bold=True)
rbox(0.6, 4.8, 1.8, 0.9, "BLIP-2\nCaptioner", C_MODEL, fs=8.5, bold=True)
rbox(0.4, 3.2, 2.2, 0.9, '"A cat sitting\n  on a table"', "#F5F5F5", fs=8,
     ec="#BDBDBD", lw=0.6)

arr(1.5, 6.4, 1.5, 5.75)
arr(1.5, 4.8, 1.5, 4.15)
note(1.85, 6.05, "image", fs=6)


# ══════════════════════ STAGE 1 (col 2) ═══════════════════════════════════
rbox(3.2, 6.2, 2.0, 1.1, "Triple Extractor\n(spaCy\ndep. parsing)", C_S1,
     fs=8.5, bold=True)

# Triple output box
rbox(3.1, 4.5, 2.2, 1.2,
     "(cat, sit, table)\n(cat, on, table)",
     "#FFF8E1", fs=8, ec="#FFB74D", lw=0.6)

# Type tags
stag(5.55, 5.2, "ACTION", "#E3F2FD", "#1565C0")
stag(5.55, 4.75, "SPATIAL", "#E8F5E9", "#2E7D32")

# caption → extractor
arr(2.6, 3.65, 4.2, 3.65)
arr(4.2, 3.65, 4.2, 4.5)
# extractor → triples
arr(4.2, 6.2, 4.2, 5.75)


# ══════════════════════ STAGE 2 (col 3) ═══════════════════════════════════

# ---- Decision diamond ----
dia(8.2, 6.5, 1.6, 0.9, "Relation\nType?", fs=7.5)

# ---- Spatial branch (left) ----
rbox(6.2, 4.7, 2.0, 0.9, "Spatial Verifier\nOWL-ViT + Geom.", C_SPAT, fs=8, bold=True)

# ---- VQA branch (right) ----
rbox(8.7, 4.7, 2.0, 0.9, "VQA Verifier\nBLIP-2 Yes/No", C_VQA, fs=8, bold=True)

# ---- Verdict ----
rbox(7.2, 2.5, 2.0, 0.9, "Verdict per triple\nverified / halluc.", "#ECEFF1",
     fs=8, ec="#90A4AE", lw=0.6)

# Triples → Decision
arr(5.3, 5.1, 7.4, 6.5)

# Decision → Spatial (left branch)
arr(7.4, 6.3, 7.2, 5.65)
ax.text(6.8, 6.1, "SPATIAL", fontsize=7, color="#2E7D32", fontweight="bold")

# Decision → VQA (right branch)
arr(9.0, 6.3, 9.7, 5.65)
ax.text(9.4, 6.15, "ACTION /\nATTRIBUTE", fontsize=6.5, color="#6A1B9A", fontweight="bold")

# Spatial → Verdict
arr(7.2, 4.7, 7.6, 3.45)

# VQA → Verdict
arr(9.7, 4.7, 8.8, 3.45)

# Spatial → VQA fallback (dashed horizontal)
arr(8.2, 5.15, 8.7, 5.15, c="#7B1FA2", lw=0.7, ls="--")
note(8.45, 4.5, "fallback if\nconf < 0.45", fs=6, c="#7B1FA2")

# Image features to verifiers (dotted)
# Route from image up and across the top
arr(1.5, 7.35, 7.2, 7.35, c="#90CAF9", lw=0.5, ls=":", sty="-")
arr(7.2, 7.35, 7.2, 5.65, c="#90CAF9", lw=0.5, ls=":")
arr(7.2, 7.35, 9.7, 7.35, c="#90CAF9", lw=0.5, ls=":", sty="-")
arr(9.7, 7.35, 9.7, 5.65, c="#90CAF9", lw=0.5, ls=":")
note(4.3, 7.55, "image features (visual grounding)", fs=6.5, c="#64B5F6")


# ══════════════════════ STAGE 3 (col 4) ═══════════════════════════════════

# Corrector
rbox(10.9, 5.9, 2.2, 1.2, "Minimal Corrector\n(Mistral-7B\nvia Together.ai)", C_S3,
     fs=8.5, bold=True)

# Self-consistency guard
rbox(11.05, 4.3, 1.9, 0.85, "Self-Consistency\nGuard", "#FFCCBC", fs=8, bold=True,
     ec="#FF8A65", lw=0.7)

# Output
rbox(10.7, 1.8, 2.6, 1.3,
     "Corrected Caption\n+ Evaluation Metrics\n(R-CHAIR, R-POPE,\n Edit Rate, BLEU-4)",
     C_OUT, fs=8, bold=True)

# Verdict → Corrector (horizontal then up)
arr(9.2, 2.95, 12.0, 2.95)
arr(12.0, 2.95, 12.0, 4.3)
note(10.6, 3.15, "hallucinated triples", fs=6)

# Corrector → Self-consistency
arr(12.0, 5.9, 12.0, 5.2)

# Self-consistency → Output
arr(12.0, 4.3, 12.0, 3.15)

# Self-consistency reject loop (left side)
ax.annotate("", xy=(10.55, 6.5), xytext=(10.55, 4.72),
            arrowprops=dict(arrowstyle="-|>", color="#D84315", lw=0.7,
                            linestyle="--", connectionstyle="arc3,rad=-0.4"),
            zorder=4)
note(10.2, 5.55, "reject\n& retry", fs=6, c="#BF360C")


# ══════════════════════ LEGEND ════════════════════════════════════════════
ly = 0.35
ax.plot([4.2, 4.7], [ly, ly], color="#90CAF9", lw=0.8, ls=":", zorder=5)
note(5.45, ly, "= image features (visual input)", fs=6, c="#64B5F6")

ax.plot([7.3, 7.8], [ly, ly], color="#7B1FA2", lw=0.8, ls="--", zorder=5)
note(8.65, ly, "= low-confidence fallback", fs=6, c="#7B1FA2")

ax.plot([10.2, 10.7], [ly, ly], color="#D84315", lw=0.8, ls="--", zorder=5)
note(11.45, ly, "= reject & retry", fs=6, c="#BF360C")


# ══════════════════════ SAVE ══════════════════════════════════════════════
plt.tight_layout(pad=0.3)
fig.savefig("architecture.png", dpi=300, bbox_inches="tight", facecolor=BG)
fig.savefig("architecture.pdf", dpi=300, bbox_inches="tight", facecolor=BG)
print("Saved: architecture.png + architecture.pdf")
plt.close()
