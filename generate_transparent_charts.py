#!/usr/bin/env python3
"""
📊 TRANSPARENT BENCHMARK CHART GENERATOR
=========================================
Generates comparison charts with CLEAR INDICATION of data sources:
  - ✓ TESTED: Empirically measured on actual Odia dataset
  - 📊 ESTIMATED: Based on published documentation
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

print("\n" + "="*80)
print("📊 GENERATING TRANSPARENT BENCHMARK VISUALIZATIONS")
print("="*80 + "\n")

# Load real benchmark data
with open("REAL_BENCHMARK_RESULTS.json") as f:
    data = json.load(f)

results = data["results"]["accuracy_metrics"]
speeds = data["results"]["speed_metrics"]
memory = data["results"]["memory_metrics"]

# Create output directory
Path("blog").mkdir(exist_ok=True)

# ============================================================================
# CHART 1: ACCURACY COMPARISON
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

models = list(results.keys())
accuracy = [results[m]["accuracy_percent"] for m in models]
tested = [results[m]["tested"] == "Yes - Phase 2A evaluation" or results[m]["tested"] == "Yes - Direct measurement (struggled with Odia)" for m in models]

colors = ["#FF6B35" if t else "#004E89" for t in tested]
markers = ["●" if t else "○" for t in tested]

bars = ax.barh(models, accuracy, color=colors, alpha=0.8, edgecolor="black", linewidth=2)

# Add value labels with markers
for i, (bar, acc, is_tested) in enumerate(zip(bars, accuracy, tested)):
    marker = "✓ TESTED" if is_tested else "📊 EST."
    label = f"{acc}% {marker}"
    ax.text(acc + 1, i, label, va='center', fontweight='bold', fontsize=11)

ax.set_xlabel("Accuracy (%)", fontsize=12, fontweight='bold')
ax.set_title("Odia OCR Model Accuracy Comparison\n(Your Model vs Alternatives)", fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, 70)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Legend
tested_patch = mpatches.Patch(color='#FF6B35', label='✓ Tested on Odia dataset')
estimated_patch = mpatches.Patch(color='#004E89', label='📊 Estimated from literature')
ax.legend(handles=[tested_patch, estimated_patch], loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig("blog/1_accuracy_comparison.png", dpi=300, bbox_inches='tight')
print("✅ Chart 1: Accuracy Comparison - saved")
plt.close()

# ============================================================================
# CHART 2: SPEED COMPARISON
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

models_speed = list(speeds.keys())
times = [speeds[m]["milliseconds_per_image"] for m in models_speed]
tested_speed = [results[m]["tested"] == "Yes - Phase 2A evaluation" or results[m]["tested"] == "Yes - Direct measurement (struggled with Odia)" for m in models_speed]

colors_speed = ["#FF6B35" if t else "#004E89" for t in tested_speed]

bars = ax.barh(models_speed, times, color=colors_speed, alpha=0.8, edgecolor="black", linewidth=2)

# Add value labels
for i, (bar, time, is_tested) in enumerate(zip(bars, times, tested_speed)):
    marker = "✓" if is_tested else "📊"
    label = f"{time}ms {marker}"
    ax.text(time + 50, i, label, va='center', fontweight='bold', fontsize=11)

ax.set_xlabel("Time per Image (milliseconds, lower is better)", fontsize=12, fontweight='bold')
ax.set_title("Odia OCR Model Speed Comparison\n(Inference time per image)", fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Legend
tested_patch = mpatches.Patch(color='#FF6B35', label='✓ Tested on Odia dataset')
estimated_patch = mpatches.Patch(color='#004E89', label='📊 Estimated from literature')
ax.legend(handles=[tested_patch, estimated_patch], loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig("blog/2_speed_comparison.png", dpi=300, bbox_inches='tight')
print("✅ Chart 2: Speed Comparison - saved")
plt.close()

# ============================================================================
# CHART 3: ACCURACY vs SPEED TRADE-OFF
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

for model in models:
    acc = results[model]["accuracy_percent"]
    spd = speeds[model]["milliseconds_per_image"]
    is_tested = results[model]["tested"] == "Yes - Phase 2A evaluation" or results[model]["tested"] == "Yes - Direct measurement (struggled with Odia)"
    
    color = "#FF6B35" if is_tested else "#004E89"
    marker = "o" if is_tested else "s"
    size = 400 if is_tested else 300
    
    ax.scatter(spd, acc, s=size, color=color, marker=marker, alpha=0.7, edgecolors='black', linewidth=2)
    
    # Add labels
    label_marker = "✓" if is_tested else "📊"
    ax.annotate(f"{model}\n{label_marker}", (spd, acc), fontsize=10, ha='center', va='bottom', fontweight='bold')

ax.set_xlabel("Speed (milliseconds, lower is better →)", fontsize=12, fontweight='bold')
ax.set_ylabel("Accuracy (%, higher is better ↑)", fontsize=12, fontweight='bold')
ax.set_title("Odia OCR: Accuracy vs Speed Trade-off\n(Find your sweet spot)", fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Legend
tested_patch = mpatches.Patch(color='#FF6B35', label='✓ Tested (circles)')
estimated_patch = mpatches.Patch(color='#004E89', label='📊 Estimated (squares)')
ax.legend(handles=[tested_patch, estimated_patch], loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig("blog/3_accuracy_vs_speed.png", dpi=300, bbox_inches='tight')
print("✅ Chart 3: Accuracy vs Speed - saved")
plt.close()

# ============================================================================
# CHART 4: MEMORY USAGE
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

models_mem = list(memory.keys())
mem_values = [memory[m]["total_gb"] for m in models_mem]
tested_mem = [results[m]["tested"] == "Yes - Phase 2A evaluation" or results[m]["tested"] == "Yes - Direct measurement (struggled with Odia)" for m in models_mem]

colors_mem = ["#FF6B35" if t else "#004E89" for t in tested_mem]

bars = ax.barh(models_mem, mem_values, color=colors_mem, alpha=0.8, edgecolor="black", linewidth=2)

# Add value labels
for i, (bar, mem, is_tested) in enumerate(zip(bars, mem_values, tested_mem)):
    marker = "✓" if is_tested else "📊"
    label = f"{mem}GB {marker}"
    ax.text(mem + 0.5, i, label, va='center', fontweight='bold', fontsize=11)

ax.set_xlabel("Total Memory (GB, lower is better)", fontsize=12, fontweight='bold')
ax.set_title("Odia OCR Model Memory Requirements\n(GPU + CPU combined)", fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Legend
tested_patch = mpatches.Patch(color='#FF6B35', label='✓ Tested on Odia dataset')
estimated_patch = mpatches.Patch(color='#004E89', label='📊 Estimated from literature')
ax.legend(handles=[tested_patch, estimated_patch], loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig("blog/4_memory_comparison.png", dpi=300, bbox_inches='tight')
print("✅ Chart 4: Memory Comparison - saved")
plt.close()

# ============================================================================
# CHART 5: OVERALL COMPARISON (RADAR CHART)
# ============================================================================

from math import pi

categories = ['Accuracy', 'Speed', 'Memory Efficiency', 'Practicality']
N = len(categories)

# Normalize metrics to 0-100 scale
def normalize_score(model_name):
    # Accuracy: 0-100
    acc = results[model_name]["accuracy_percent"]
    
    # Speed: reverse scale (faster = higher score)
    max_speed = max([speeds[m]["milliseconds_per_image"] for m in speeds.keys()])
    speed_score = (1 - speeds[model_name]["milliseconds_per_image"] / max_speed) * 100
    
    # Memory: reverse scale (less = higher score)
    max_mem = max([memory[m]["total_gb"] for m in memory.keys()])
    mem_score = (1 - memory[model_name]["total_gb"] / max_mem) * 100
    
    # Practicality: balance of all three
    practicality = (acc + speed_score + mem_score) / 3
    
    return [acc, speed_score, mem_score, practicality]

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Plot models
for model in models:
    scores = normalize_score(model)
    scores += scores[:1]
    
    is_tested = results[model]["tested"] == "Yes - Phase 2A evaluation" or results[model]["tested"] == "Yes - Direct measurement (struggled with Odia)"
    color = "#FF6B35" if is_tested else "#004E89"
    linewidth = 3 if is_tested else 2
    linestyle = '-' if is_tested else '--'
    marker = "o" if is_tested else "s"
    
    ax.plot(angles, scores, linewidth=linewidth, label=model, marker=marker, linestyle=linestyle, color=color, alpha=0.7)
    ax.fill(angles, scores, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
ax.grid(True, linestyle='--', alpha=0.7)

plt.title("Overall Model Comparison Profile\n(Higher scores = Better, ✓ = Tested)", 
          fontsize=14, fontweight='bold', pad=20)

# Legend
tested_line = plt.Line2D([0], [0], color='#FF6B35', linewidth=3, label='✓ Tested')
estimated_line = plt.Line2D([0], [0], color='#004E89', linewidth=2, linestyle='--', label='📊 Estimated')
plt.legend(handles=[tested_line, estimated_line], loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig("blog/5_overall_comparison_radar.png", dpi=300, bbox_inches='tight')
print("✅ Chart 5: Overall Comparison - saved")
plt.close()

# ============================================================================
# CHART 6: DATA SOURCE TRANSPARENCY
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Create table showing testing methodology
table_data = []
for model in models:
    tested = results[model]["tested"]
    source = data["testing_status"][model]["source"]
    confidence = results[model]["confidence"]
    
    status = "✓ TESTED" if "Yes" in tested else "📊 ESTIMATED"
    table_data.append([model, status, source, confidence])

table = ax.table(cellText=table_data,
                colLabels=['Model', 'Status', 'Data Source', 'Confidence'],
                cellLoc='left',
                loc='center',
                colWidths=[0.18, 0.15, 0.45, 0.22])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color header
for i in range(4):
    table[(0, i)].set_facecolor('#333333')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
for i in range(1, len(table_data) + 1):
    status = table_data[i-1][1]
    color = "#FFE5D3" if "✓" in status else "#D3E5FF"
    for j in range(4):
        table[(i, j)].set_facecolor(color)

ax.axis('off')
ax.set_title("Benchmark Methodology: Data Source Transparency\n(How each metric was determined)", 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("blog/6_methodology_transparency.png", dpi=300, bbox_inches='tight')
print("✅ Chart 6: Methodology Transparency - saved")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ BENCHMARK VISUALIZATIONS GENERATED")
print("="*80)
print("\n📊 Charts created:")
print("  1. 1_accuracy_comparison.png - Model accuracy (% correct)")
print("  2. 2_speed_comparison.png - Inference speed per image")
print("  3. 3_accuracy_vs_speed.png - Trade-off analysis")
print("  4. 4_memory_comparison.png - Memory requirements")
print("  5. 5_overall_comparison_radar.png - Multi-metric radar chart")
print("  6. 6_methodology_transparency.png - Data source breakdown")

print("\n🎯 Key Transparency Features:")
print("  ✓ All tested metrics clearly marked with ✓ symbol")
print("  ✓ All estimated metrics clearly marked with 📊 symbol")
print("  ✓ Color coding: Orange (tested) vs Blue (estimated)")
print("  ✓ Separate methodology chart showing data sources")

print("\n💡 For Your Blog:")
print("  Use these charts with the caption:")
print("  'Your Qwen model (✓) vs published specs for other models (📊)'")

print("\n" + "="*80)
