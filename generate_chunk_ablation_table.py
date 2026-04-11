from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


METRICS = {
    "Hit@5": ("A_Hit", "B_Hit", True),
    "MRR@5": ("A_MRR", "B_MRR", False),
    "CPrec": ("A_CPrec", "B_CPrec", False),
    "Faith": ("A_Faith", "B_Faith", False),
    "Rel": ("A_Rel", "B_Rel", False),
    "ROUGE-L": ("A_ROUGE_L", "B_ROUGE_L", False),
}


def format_metric(value: float, is_percentage: bool, metric_name: str) -> str:
    if is_percentage:
        return f"{value * 100:.1f}%"

    precision = 4 if metric_name == "ROUGE-L" else 2
    return f"{value:.{precision}f}"


def summarize_single_file(csv_path: Path, chunk_size: int) -> list[dict[str, str | int]]:
    df = pd.read_csv(csv_path)
    rows: list[dict[str, str | int]] = []

    for method_name, suffix in (("Method A", "A"), ("Method B", "B")):
        row: dict[str, str | int] = {
            "Chunk Size": chunk_size,
            "Method": method_name,
            "Samples": len(df),
        }

        for metric_name, (_, _, is_percentage) in METRICS.items():
            col = f"{suffix}_{metric_name.replace('@5', '').replace('-', '_')}"
            if metric_name == "Hit@5":
                col = f"{suffix}_Hit"
            elif metric_name == "MRR@5":
                col = f"{suffix}_MRR"
            elif metric_name == "CPrec":
                col = f"{suffix}_CPrec"
            elif metric_name == "Faith":
                col = f"{suffix}_Faith"
            elif metric_name == "Rel":
                col = f"{suffix}_Rel"
            elif metric_name == "ROUGE-L":
                col = f"{suffix}_ROUGE_L"

            row[metric_name] = format_metric(df[col].mean(), is_percentage, metric_name)

        rows.append(row)

    return rows


def build_table(records_dir: Path, chunk_sizes: Iterable[int]) -> tuple[pd.DataFrame, list[int]]:
    all_rows: list[dict[str, str | int]] = []
    missing_chunks: list[int] = []

    for chunk_size in chunk_sizes:
        csv_path = records_dir / f"eval_Qwen3.5-2B_chunk{chunk_size}.csv"
        if not csv_path.exists():
            missing_chunks.append(chunk_size)
            continue

        all_rows.extend(summarize_single_file(csv_path, chunk_size))

    if not all_rows:
        raise FileNotFoundError("没有找到任何已完成实验的 CSV 文件。")

    table_df = pd.DataFrame(all_rows)
    table_df.sort_values(by=["Chunk Size", "Method"], inplace=True)
    return table_df, missing_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总 Chunk Size 消融实验（200/400/600/800）的大盘指标到表格。"
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=Path("记录"),
        help="实验结果 CSV 所在目录（默认：记录）",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[200, 400, 600, 800],
        help="要汇总的 Chunk Size 列表（默认：200 400 600 800）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("记录/chunk_size_ablation_summary.csv"),
        help="输出汇总 CSV 路径（默认：记录/chunk_size_ablation_summary.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    table_df, missing_chunks = build_table(args.records_dir, args.chunk_sizes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(args.output, index=False)

    print("\n=== Chunk Size 消融汇总表（已完成实验）===")
    print(table_df.to_string(index=False))
    print(f"\n汇总 CSV 已保存到: {args.output}")

    if missing_chunks:
        missing_text = ", ".join(str(c) for c in missing_chunks)
        print(f"\n[提示] 以下 Chunk Size 还未找到结果文件，待跑完后重跑脚本即可自动补齐: {missing_text}")


if __name__ == "__main__":
    main()
