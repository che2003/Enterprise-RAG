import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 统一语义配色：Method A=蓝，Method B=橙
COLOR_A = "#1f77b4"
COLOR_B = "#ff7f0e"
GRID_COLOR = "#e6e6e6"

METRICS = {
    "Hit@5": ("A_Hit", "B_Hit"),
    "MRR@5": ("A_MRR", "B_MRR"),
    "Context Precision": ("A_CPrec", "B_CPrec"),
    "Faithfulness": ("A_Faith", "B_Faith"),
    "Relevance": ("A_Rel", "B_Rel"),
    "ROUGE-L": ("A_ROUGE_L", "B_ROUGE_L"),
}


def _apply_paper_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "font.size": 11,
        }
    )


def _style_axis(ax, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _find_latest_csv(record_dir: str, chunk_size: int, mode: str) -> str:
    pattern = os.path.join(record_dir, f"eval_*_chunk{chunk_size}_mode-{mode}.csv")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")
    return max(candidates, key=os.path.getmtime)


def _load_mean_metrics(csv_path: str) -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(csv_path)
    means = {}
    for metric_name, (col_a, col_b) in METRICS.items():
        means[metric_name] = (float(df[col_a].mean()), float(df[col_b].mean()))
    return means


def plot_figure2_grouped_dot(csv_path: str, output_path: str) -> None:
    means = _load_mean_metrics(csv_path)
    labels = list(METRICS.keys())
    values_a = [means[label][0] for label in labels]
    values_b = [means[label][1] for label in labels]

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.hlines(y, values_a, values_b, color="#c7c7c7", linewidth=1.6, zorder=1)
    ax.scatter(values_a, y, color=COLOR_A, label="Method A", s=42, zorder=2)
    ax.scatter(values_b, y, color=COLOR_B, label="Method B", s=42, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Score")
    ax.set_title("Figure 2: Method Comparison by Metric")

    _style_axis(ax, grid_axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_figure3_chunk_sensitivity(
    record_dir: str, chunk_sizes: List[int], mode: str, output_path: str
) -> None:
    a_mrr, b_mrr, a_rouge, b_rouge = [], [], [], []

    for chunk in chunk_sizes:
        csv_path = _find_latest_csv(record_dir, chunk, mode)
        means = _load_mean_metrics(csv_path)
        a_mrr.append(means["MRR@5"][0])
        b_mrr.append(means["MRR@5"][1])
        a_rouge.append(means["ROUGE-L"][0])
        b_rouge.append(means["ROUGE-L"][1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)

    axes[0].plot(chunk_sizes, a_mrr, marker="o", color=COLOR_A, linewidth=2, label="Method A")
    axes[0].plot(chunk_sizes, b_mrr, marker="o", color=COLOR_B, linewidth=2, label="Method B")
    axes[0].set_title("MRR@5 vs Chunk Size")
    axes[0].set_xlabel("Chunk Size")
    axes[0].set_ylabel("MRR@5")
    axes[0].set_xticks(chunk_sizes)
    _style_axis(axes[0], grid_axis="y")

    axes[1].plot(chunk_sizes, a_rouge, marker="o", color=COLOR_A, linewidth=2, label="Method A")
    axes[1].plot(chunk_sizes, b_rouge, marker="o", color=COLOR_B, linewidth=2, label="Method B")
    axes[1].set_title("ROUGE-L vs Chunk Size")
    axes[1].set_xlabel("Chunk Size")
    axes[1].set_ylabel("ROUGE-L")
    axes[1].set_xticks(chunk_sizes)
    _style_axis(axes[1], grid_axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2)
    fig.suptitle("Figure 3: Chunk Size Sensitivity", y=1.08)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_figure4_retrieval_trends(record_dir: str, chunk_size: int, output_path: str) -> None:
    retrieval_modes = ["dense", "sparse", "hybrid"]
    retrieval_labels = ["Dense-only", "Sparse-only", "Hybrid-RRF"]

    a_hit, b_hit, a_mrr, b_mrr = [], [], [], []
    for mode in retrieval_modes:
        csv_path = _find_latest_csv(record_dir, chunk_size, mode)
        means = _load_mean_metrics(csv_path)
        a_hit.append(means["Hit@5"][0])
        b_hit.append(means["Hit@5"][1])
        a_mrr.append(means["MRR@5"][0])
        b_mrr.append(means["MRR@5"][1])

    x = np.arange(len(retrieval_labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)

    axes[0].plot(x, a_hit, marker="o", color=COLOR_A, linewidth=2, label="Method A")
    axes[0].plot(x, b_hit, marker="o", color=COLOR_B, linewidth=2, label="Method B")
    axes[0].set_title("Hit@5 across Retrieval Settings")
    axes[0].set_ylabel("Hit@5")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(retrieval_labels)
    _style_axis(axes[0], grid_axis="y")

    axes[1].plot(x, a_mrr, marker="o", color=COLOR_A, linewidth=2, label="Method A")
    axes[1].plot(x, b_mrr, marker="o", color=COLOR_B, linewidth=2, label="Method B")
    axes[1].set_title("MRR@5 across Retrieval Settings")
    axes[1].set_ylabel("MRR@5")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(retrieval_labels)
    _style_axis(axes[1], grid_axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2)
    fig.suptitle("Figure 4: Retrieval Ablation", y=1.08)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _apply_paper_style()

    parser = argparse.ArgumentParser(description="根据 record/*.csv 生成论文图 2/3/4（简洁论文风格）")
    parser.add_argument("--record_dir", type=str, default="record", help="评测 CSV 所在目录")
    parser.add_argument("--output_dir", type=str, default="record/figures", help="图片输出目录")
    parser.add_argument("--figure2_chunk_size", type=int, default=400, help="图2使用的 chunk size")
    parser.add_argument(
        "--figure2_mode", type=str, default="hybrid", choices=["hybrid", "dense", "sparse"], help="图2使用的检索模式"
    )
    parser.add_argument("--figure3_mode", type=str, default="hybrid", choices=["hybrid", "dense", "sparse"])
    parser.add_argument("--figure3_chunk_sizes", type=int, nargs="+", default=[200, 400, 600, 800])
    parser.add_argument("--figure4_chunk_size", type=int, default=400, help="图4使用的 chunk size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    figure2_csv = _find_latest_csv(args.record_dir, args.figure2_chunk_size, args.figure2_mode)
    figure2_out = os.path.join(args.output_dir, "figure2_method_comparison.png")
    figure3_out = os.path.join(args.output_dir, "figure3_chunk_sensitivity.png")
    figure4_out = os.path.join(args.output_dir, "figure4_retrieval_trends.png")

    plot_figure2_grouped_dot(figure2_csv, figure2_out)
    plot_figure3_chunk_sensitivity(args.record_dir, args.figure3_chunk_sizes, args.figure3_mode, figure3_out)
    plot_figure4_retrieval_trends(args.record_dir, args.figure4_chunk_size, figure4_out)

    print("图像已生成:")
    print(f"- {figure2_out}")
    print(f"- {figure3_out}")
    print(f"- {figure4_out}")


if __name__ == "__main__":
    main()
