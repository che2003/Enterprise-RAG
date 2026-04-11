import argparse
import glob
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = {
    "Hit@5": ("A_Hit", "B_Hit"),
    "MRR@5": ("A_MRR", "B_MRR"),
    "Context Precision": ("A_CPrec", "B_CPrec"),
    "Faithfulness": ("A_Faith", "B_Faith"),
    "Relevance": ("A_Rel", "B_Rel"),
    "ROUGE-L": ("A_ROUGE_L", "B_ROUGE_L"),
}


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


def _min_max_normalize(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=float)
    v_min = arr.min()
    v_max = arr.max()
    if np.isclose(v_max, v_min):
        return [1.0 for _ in values]
    return ((arr - v_min) / (v_max - v_min)).tolist()


def plot_radar_chart(csv_path: str, output_path: str) -> None:
    means = _load_mean_metrics(csv_path)

    labels = list(METRICS.keys())
    raw_a = [means[label][0] for label in labels]
    raw_b = [means[label][1] for label in labels]

    norm_a = _min_max_normalize(raw_a)
    norm_b = _min_max_normalize(raw_b)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    norm_a += norm_a[:1]
    norm_b += norm_b[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, norm_a, linewidth=2, label="Method A")
    ax.fill(angles, norm_a, alpha=0.2)
    ax.plot(angles, norm_b, linewidth=2, label="Method B")
    ax.fill(angles, norm_b, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Figure 2: Method A vs Method B Radar (Normalized)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_chunk_sensitivity(record_dir: str, chunk_sizes: List[int], mode: str, output_path: str) -> None:
    a_mrr, b_mrr, a_rouge, b_rouge = [], [], [], []

    for chunk in chunk_sizes:
        csv_path = _find_latest_csv(record_dir, chunk, mode)
        means = _load_mean_metrics(csv_path)
        a_mrr.append(means["MRR@5"][0])
        b_mrr.append(means["MRR@5"][1])
        a_rouge.append(means["ROUGE-L"][0])
        b_rouge.append(means["ROUGE-L"][1])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    line1 = ax1.plot(chunk_sizes, a_mrr, marker="o", linewidth=2, label="Method A - MRR@5")
    line2 = ax1.plot(chunk_sizes, b_mrr, marker="o", linewidth=2, label="Method B - MRR@5")

    line3 = ax2.plot(chunk_sizes, a_rouge, marker="s", linestyle="--", linewidth=2, label="Method A - ROUGE-L")
    line4 = ax2.plot(chunk_sizes, b_rouge, marker="s", linestyle="--", linewidth=2, label="Method B - ROUGE-L")

    ax1.set_xlabel("Chunk Size")
    ax1.set_ylabel("MRR@5")
    ax2.set_ylabel("ROUGE-L")
    ax1.set_title("Figure 3: Chunk Size Sensitivity")
    ax1.set_xticks(chunk_sizes)

    all_lines = line1 + line2 + line3 + line4
    all_labels = [line.get_label() for line in all_lines]
    ax1.legend(all_lines, all_labels, loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_retrieval_ablation(record_dir: str, chunk_size: int, output_path: str) -> None:
    retrieval_modes = ["dense", "sparse", "hybrid"]

    a_hit, b_hit, a_mrr, b_mrr = [], [], [], []
    for mode in retrieval_modes:
        csv_path = _find_latest_csv(record_dir, chunk_size, mode)
        means = _load_mean_metrics(csv_path)
        a_hit.append(means["Hit@5"][0])
        b_hit.append(means["Hit@5"][1])
        a_mrr.append(means["MRR@5"][0])
        b_mrr.append(means["MRR@5"][1])

    x = np.arange(len(retrieval_modes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, a_hit, width, label="A Hit@5")
    ax.bar(x - 0.5 * width, a_mrr, width, label="A MRR@5")
    ax.bar(x + 0.5 * width, b_hit, width, label="B Hit@5")
    ax.bar(x + 1.5 * width, b_mrr, width, label="B MRR@5")

    ax.set_xticks(x)
    ax.set_xticklabels(["Dense-only", "Sparse-only", "Hybrid-RRF"])
    ax.set_ylabel("Score")
    ax.set_title("Figure 4: Retrieval Ablation (Dense/Sparse/Hybrid-RRF)")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据 record/*.csv 生成论文图 2/3/4")
    parser.add_argument("--record_dir", type=str, default="record", help="评测 CSV 所在目录")
    parser.add_argument("--output_dir", type=str, default="record/figures", help="图片输出目录")
    parser.add_argument("--radar_chunk_size", type=int, default=400, help="图2使用的 chunk size")
    parser.add_argument("--radar_mode", type=str, default="hybrid", choices=["hybrid", "dense", "sparse"])
    parser.add_argument("--line_mode", type=str, default="hybrid", choices=["hybrid", "dense", "sparse"])
    parser.add_argument("--line_chunk_sizes", type=int, nargs="+", default=[200, 400, 600, 800])
    parser.add_argument("--ablation_chunk_size", type=int, default=400, help="图4使用的 chunk size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    radar_csv = _find_latest_csv(args.record_dir, args.radar_chunk_size, args.radar_mode)
    radar_out = os.path.join(args.output_dir, "figure2_radar.png")
    line_out = os.path.join(args.output_dir, "figure3_chunk_sensitivity.png")
    ablation_out = os.path.join(args.output_dir, "figure4_retrieval_ablation.png")

    plot_radar_chart(radar_csv, radar_out)
    plot_chunk_sensitivity(args.record_dir, args.line_chunk_sizes, args.line_mode, line_out)
    plot_retrieval_ablation(args.record_dir, args.ablation_chunk_size, ablation_out)

    print("图像已生成:")
    print(f"- {radar_out}")
    print(f"- {line_out}")
    print(f"- {ablation_out}")


if __name__ == "__main__":
    main()
