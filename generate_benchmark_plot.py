"""
Generate benchmark plots for the HF README:
  1. CER & Accuracy vs Checkpoint (dual-axis line chart)
  2. Per-category Accuracy comparison: best ckpt-1300 vs latest ckpt-1800 (grouped bar chart)

Run with: HF_TOKEN=xxx python3 generate_benchmark_plot.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from huggingface_hub import HfApi

# ── Benchmark data ────────────────────────────────────────────────────────────
data = [
    {"step": 300,  "cer": 0.902, "acc": 9.9},
    {"step": 900,  "cer": 0.804, "acc": 19.6},
    {"step": 1000, "cer": 0.863, "acc": 13.7},
    {"step": 1300, "cer": 0.655, "acc": 34.5},   # best
    {"step": 1400, "cer": 0.690, "acc": 31.0},
    {"step": 1500, "cer": 0.690, "acc": 31.0},
    {"step": 1600, "cer": 0.758, "acc": 24.2},
    {"step": 1700, "cer": 0.912, "acc": 8.8},
    {"step": 1800, "cer": 0.750, "acc": 24.9},
]

steps = [d["step"] for d in data]
cers  = [d["cer"]  for d in data]
accs  = [d["acc"]  for d in data]

best_idx = accs.index(max(accs))
best_step = steps[best_idx]

# ── Per-category data (accuracy = (1-CER)*100, clipped to 0) ─────────────────
CATEGORIES = ["handwritten", "scene_text", "Digital", "printed", "Book", "Newspaper"]
cat_display = ["Handwritten", "Scene Text", "Digital", "Printed", "Book", "Newspaper"]

# ckpt-1300 (best overall)
cat_cer_1300 = {
    "handwritten": 0.377,
    "scene_text":  0.472,
    "Digital":     0.601,
    "printed":     0.844,
    "Book":        0.891,
    "Newspaper":   0.923,
}
# ckpt-1800 (latest)
cat_cer_1800 = {
    "handwritten": 0.385,
    "scene_text":  0.552,
    "Digital":     0.628,
    "printed":     1.040,
    "Book":        0.899,
    "Newspaper":   0.930,
}

acc_1300 = [max(0.0, (1 - cat_cer_1300[c]) * 100) for c in CATEGORIES]
acc_1800 = [max(0.0, (1 - cat_cer_1800[c]) * 100) for c in CATEGORIES]

# =====================================================================
# Plot 1 — CER & Accuracy vs Checkpoint
# =====================================================================
fig, ax1 = plt.subplots(figsize=(11, 5.5))
ax2 = ax1.twinx()

# CER line (lower is better)
l1, = ax1.plot(steps, cers, "o-", color="#E05C5C", linewidth=2.5,
               markersize=7, label="CER (↓ lower is better)")
ax1.set_ylabel("Character Error Rate (CER)", color="#C03030", fontsize=12)
ax1.tick_params(axis="y", labelcolor="#C03030")
ax1.set_ylim(0.0, 1.1)

# Accuracy line (higher is better)
l2, = ax2.plot(steps, accs, "s--", color="#3A86FF", linewidth=2.5,
               markersize=7, label="Accuracy % (↑ higher is better)")
ax2.set_ylabel("Accuracy (1 − CER) %", color="#1A66CF", fontsize=12)
ax2.tick_params(axis="y", labelcolor="#1A66CF")
ax2.set_ylim(0, 55)

# Shade overfitting region
ax1.axvspan(best_step, max(steps) + 50, alpha=0.08, color="#E05C5C")
ax1.text(best_step + 60, 1.02, "⚠ Overfitting zone",
         color="#C03030", fontsize=9, fontstyle="italic")

# Best checkpoint line
ax1.axvline(x=best_step, color="#2DC653", linewidth=2.2, linestyle="-.",
            label=f"Best checkpoint (step {best_step})")
ax1.annotate(f"★ Best\nCER={cers[best_idx]:.3f}\nAcc={accs[best_idx]:.1f}%",
             xy=(best_step, cers[best_idx]),
             xytext=(best_step + 90, cers[best_idx] - 0.12),
             arrowprops=dict(arrowstyle="->", color="#2DC653"),
             color="#2DC653", fontsize=9.5, fontweight="bold")

# Mark each point with its acc value
for s, c, a in zip(steps, cers, accs):
    ax2.annotate(f"{a:.0f}%", xy=(s, a), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8,
                color="#1A66CF")

ax1.set_xlabel("Training Step (Checkpoint)", fontsize=12)
ax1.set_xticks(steps)
ax1.set_xticklabels([str(s) for s in steps], rotation=30, ha="right")

plt.title(
    "Odia OCR — Qwen2.5-VL-7B LoRA: Benchmark CER & Accuracy vs Checkpoint\n"
    "(Evaluated on Iftesha/odia-ocr-benchmark, 151 samples)",
    fontsize=12, fontweight="bold", pad=12
)

handles = [l1, l2,
           mpatches.Patch(color="#2DC653", label=f"Best: ckpt-{best_step} (Acc={accs[best_idx]:.1f}%)")]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.92),
           fontsize=9, framealpha=0.92)

fig.tight_layout()
out_path1 = "/tmp/benchmark_cer_accuracy.png"
plt.savefig(out_path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plot 1 to {out_path1}")

# =====================================================================
# Plot 2 — Per-category Accuracy: ckpt-1300 (best) vs ckpt-1800 (latest)
# =====================================================================
fig2, ax = plt.subplots(figsize=(11, 5.5))

x = np.arange(len(CATEGORIES))
width = 0.35

bars1 = ax.bar(x - width/2, acc_1300, width, label="ckpt-1300 [Best]",
               color="#2DC653", edgecolor="white", linewidth=0.8)
bars2 = ax.bar(x + width/2, acc_1800, width, label="ckpt-1800 (Latest)",
               color="#3A86FF", edgecolor="white", linewidth=0.8)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
            color="#1A7D3A", fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
            color="#1A5FAF", fontweight="bold")

ax.set_xlabel("OCR Category", fontsize=12)
ax.set_ylabel("Accuracy — (1 − CER) %", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(cat_display, fontsize=11)
ax.set_ylim(0, 85)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

ax.legend(fontsize=10, loc="upper right")
plt.title(
    "Odia OCR — Per-category Accuracy: Best Checkpoint vs Latest\n"
    "(ckpt-1300 best overall vs ckpt-1800 latest  |  Iftesha/odia-ocr-benchmark)",
    fontsize=12, fontweight="bold", pad=12
)

fig2.tight_layout()
out_path2 = "/tmp/benchmark_category_accuracy.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plot 2 to {out_path2}")

# ── Upload to HF ─────────────────────────────────────────────────────────────
token = os.environ.get("HF_TOKEN")
if not token:
    print("No HF_TOKEN — skipping upload")
else:
    api = HfApi(token=token)
    for local, remote in [
        (out_path1, "assets/benchmark_cer_accuracy.png"),
        (out_path2, "assets/benchmark_category_accuracy.png"),
    ]:
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id="shantipriya/odia-ocr-qwen-finetuned_v3",
            repo_type="model",
            commit_message=f"Update benchmark plots: {os.path.basename(remote)}",
        )
        print(f"Uploaded to HF: {remote}")
