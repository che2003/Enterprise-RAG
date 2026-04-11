from __future__ import annotations

import argparse
import re
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

CHUNK_PATTERN = re.compile(r"chunk(\d+)", re.IGNORECASE)


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
            "Source File": str(csv_path),
        }

        for metric_name, (_, _, is_percentage) in METRICS.items():
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
            else:
                col = f"{suffix}_ROUGE_L"

            row[metric_name] = format_metric(df[col].mean(), is_percentage, metric_name)

        rows.append(row)

    return rows


def extract_chunk_size(path: Path) -> int | None:
    match = CHUNK_PATTERN.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def find_eval_files(records_dirs: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for records_dir in records_dirs:
        if not records_dir.exists():
            continue
        found.extend(records_dir.rglob("eval_*chunk*.csv"))
    return sorted(set(found))


def build_table(records_dirs: list[Path], chunk_sizes: Iterable[int]) -> tuple[pd.DataFrame, list[int], dict[int, Path]]:
    target_sizes = set(chunk_sizes)
    file_map: dict[int, Path] = {}

    for csv_path in find_eval_files(records_dirs):
        chunk_size = extract_chunk_size(csv_path)
        if chunk_size is None or chunk_size not in target_sizes:
            continue

        # 同一个 chunk size 若匹配到多个文件，保留最新修改时间的文件
        if chunk_size not in file_map or csv_path.stat().st_mtime > file_map[chunk_size].stat().st_mtime:
            file_map[chunk_size] = csv_path

    all_rows: list[dict[str, str | int]] = []
    missing_chunks: list[int] = []

    for chunk_size in sorted(target_sizes):
        if chunk_size not in file_map:
            missing_chunks.append(chunk_size)
            continue
        all_rows.extend(summarize_single_file(file_map[chunk_size], chunk_size))

    if not all_rows:
        search_text = ", ".join(str(p) for p in records_dirs)
        raise FileNotFoundError(f"没有在这些目录找到可用 CSV: {search_text}")

    table_df = pd.DataFrame(all_rows)
    table_df.sort_values(by=["Chunk Size", "Method"], inplace=True)
    return table_df, missing_chunks, file_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总 Chunk Size 消融实验（200/400/600/800）的大盘指标到表格。"
    )
    parser.add_argument(
        "--records-dirs",
        nargs="+",
        type=Path,
        default=[Path("记录"), Path("record"), Path(".")],
        help="要搜索 CSV 的目录列表（默认：记录 record 当前目录）",
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

    table_df, missing_chunks, file_map = build_table(args.records_dirs, args.chunk_sizes)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(args.output, index=False)

    print("\n=== Chunk Size 消融汇总表（已完成实验）===")
    print(table_df.to_string(index=False))
    print(f"\n汇总 CSV 已保存到: {args.output}")

    print("\n=== 已匹配到的输入文件 ===")
    for chunk_size in sorted(file_map):
        print(f"chunk {chunk_size}: {file_map[chunk_size]}")

    if missing_chunks:
        missing_text = ", ".join(str(c) for c in missing_chunks)
        print(f"\n[提示] 以下 Chunk Size 还未找到结果文件，待跑完后重跑脚本即可自动补齐: {missing_text}")


if __name__ == "__main__":
    main()
