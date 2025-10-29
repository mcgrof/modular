# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

#!/usr/bin/env python3
"""
Generate performance visualization for AMR benchmark results.
Compares Mojo compiler optimizations against C++ baseline.
"""

import matplotlib.pyplot as plt
import numpy as np

# Performance data from actual benchmark runs
# All values in Mcells/sec

mojo_structured = {
    "Auto-vectorization": 5.02,
    "Explicit SIMD": 69.42,
    "Cache blocking": 724.47,
}

mojo_unstructured = {
    "Auto-vectorization": 4.68,
    "SIMD + @parameter": 5.12,
    "prefetch + inline": 5.14,
}

cpp_performance = {
    "Structured (direct)": 65.96,
    "Unstructured (CSR)": 100.87,
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# === Plot 1: Structured Mesh Performance ===
categories_structured = list(mojo_structured.keys())
mojo_values_structured = list(mojo_structured.values())
cpp_structured_value = cpp_performance["Structured (direct)"]

x_pos = np.arange(len(categories_structured))
bars = ax1.bar(
    x_pos,
    mojo_values_structured,
    color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)

# Add C++ baseline as horizontal line
ax1.axhline(
    y=cpp_structured_value,
    color="red",
    linestyle="--",
    linewidth=2.5,
    label=f"C++ GCC 15.2 -O3 ({cpp_structured_value:.2f} Mcells/sec)",
)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, mojo_values_structured, strict=False)):
    height = bar.get_height()
    speedup = val / mojo_values_structured[0]
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 20,
        f"{val:.2f}\n({speedup:.1f}x)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax1.set_ylabel("Throughput (Mcells/sec)", fontsize=13, fontweight="bold")
ax1.set_title(
    "Structured Mesh: Mojo Compiler Optimizations\n(64×64 grid, 100 timesteps)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(categories_structured, rotation=15, ha="right")
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_ylim(0, 800)

# Add annotation for best result
ax1.annotate(
    "11.0x faster\nthan C++!",
    xy=(2, 724.47),
    xytext=(1.5, 600),
    fontsize=12,
    fontweight="bold",
    color="green",
    arrowprops=dict(arrowstyle="->", color="green", lw=2),
)

# === Plot 2: Unstructured Mesh Performance ===
categories_unstructured = list(mojo_unstructured.keys())
mojo_values_unstructured = list(mojo_unstructured.values())
cpp_unstructured_value = cpp_performance["Unstructured (CSR)"]

x_pos2 = np.arange(len(categories_unstructured))
bars2 = ax2.bar(
    x_pos2,
    mojo_values_unstructured,
    color=["#1f77b4", "#ff7f0e", "#9467bd"],
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)

# Add C++ baseline as horizontal line
ax2.axhline(
    y=cpp_unstructured_value,
    color="red",
    linestyle="--",
    linewidth=2.5,
    label=f"C++ GCC 15.2 -O3 ({cpp_unstructured_value:.2f} Mcells/sec)",
)

# Add value labels on bars
for i, (bar, val) in enumerate(
    zip(bars2, mojo_values_unstructured, strict=False)
):
    height = bar.get_height()
    speedup = val / mojo_values_unstructured[0]
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 3,
        f"{val:.2f}\n({speedup:.2f}x)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax2.set_ylabel("Throughput (Mcells/sec)", fontsize=13, fontweight="bold")
ax2.set_title(
    "Unstructured Mesh (CSR): Indirect Memory Access\n(64×64 grid, 100 timesteps)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(categories_unstructured, rotation=15, ha="right")
ax2.legend(fontsize=11, loc="upper right")
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_ylim(0, 120)

# Add annotation for C++ advantage
ax2.annotate(
    "C++ 20x faster\n(optimization opportunity)",
    xy=(0.5, cpp_unstructured_value),
    xytext=(1.2, 70),
    fontsize=11,
    fontweight="bold",
    color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred", lw=2),
)

# Overall title
fig.suptitle(
    "AMR Benchmark: Mojo Compiler Features vs C++ (GCC 15.2)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# Add footer text
footer_text = (
    "Mojo features: @parameter (loop unrolling) • SIMD[DType, width] (explicit vectorization) • "
    "@always_inline • prefetch() • alias (compile-time math) • parametric tile_size"
)
fig.text(
    0.5,
    0.02,
    footer_text,
    ha="center",
    fontsize=10,
    style="italic",
    wrap=True,
    color="#555555",
)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig("amr_benchmark_results.png", dpi=150, bbox_inches="tight")
print("✓ Generated: amr_benchmark_results.png")

# === Create a second plot: Speedup comparison ===
fig2, ax = plt.subplots(figsize=(12, 7))

# Prepare data for grouped bar chart
optimization_levels = [
    "Baseline\n(auto-vec)",
    "Level 1\n(SIMD/prefetch)",
    "Level 2\n(cache blocking)",
]

mojo_struct_speedups = [
    1.0,  # baseline
    69.42 / 5.02,  # explicit SIMD
    724.47 / 5.02,  # cache blocking
]

mojo_unstruct_speedups = [
    1.0,  # baseline
    5.14 / 4.68,  # prefetch + inline
    None,  # no cache blocking for unstructured
]

cpp_struct_speedup = (
    cpp_structured_value / mojo_structured["Auto-vectorization"]
)
cpp_unstruct_speedup = (
    cpp_unstructured_value / mojo_unstructured["Auto-vectorization"]
)

# Bar positions
x = np.arange(len(optimization_levels))
width = 0.35

# Plot bars
bars1 = ax.bar(
    x - width / 2,
    mojo_struct_speedups[: len(x)],
    width,
    label="Mojo Structured",
    color="#2ca02c",
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)
bars2 = ax.bar(
    x + width / 2,
    [mojo_unstruct_speedups[0], mojo_unstruct_speedups[1], 0],
    width,
    label="Mojo Unstructured",
    color="#9467bd",
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)

# Add C++ reference lines
ax.axhline(
    y=cpp_struct_speedup,
    color="darkgreen",
    linestyle=":",
    linewidth=2,
    alpha=0.7,
    label=f"C++ Structured ({cpp_struct_speedup:.1f}x)",
)
ax.axhline(
    y=cpp_unstruct_speedup,
    color="purple",
    linestyle=":",
    linewidth=2,
    alpha=0.7,
    label=f"C++ Unstructured ({cpp_unstruct_speedup:.1f}x)",
)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}x",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

ax.set_ylabel("Speedup vs Baseline (log scale)", fontsize=13, fontweight="bold")
ax.set_title(
    "Mojo Compiler Optimization Impact: Speedup Over Baseline\nComparison with C++ GCC 15.2 -O3 -march=native",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xticks(x)
ax.set_xticklabels(optimization_levels)
ax.legend(fontsize=11, loc="upper left")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_yscale("log")
ax.set_ylim(0.5, 200)

# Add key insight text box
textstr = (
    "Key Insights:\n"
    "• Cache blocking (structured): 144x speedup\n"
    "• Mojo beats C++ by 11x with explicit control\n"
    "• Unstructured: optimization opportunity"
)
props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
ax.text(
    0.98,
    0.97,
    textstr,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=props,
)

plt.tight_layout()
plt.savefig("amr_speedup_comparison.png", dpi=150, bbox_inches="tight")
print("✓ Generated: amr_speedup_comparison.png")

print("\nBenchmark visualizations created successfully!")
print("  - amr_benchmark_results.png (throughput comparison)")
print("  - amr_speedup_comparison.png (speedup analysis)")
