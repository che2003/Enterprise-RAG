import pandas as pd


def generate_summary_from_csv(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)

        metrics = {
            "Hit@5": ("A_Hit", "B_Hit", True),
            "MRR@5": ("A_MRR", "B_MRR", False),
            "CPrec": ("A_CPrec", "B_CPrec", False),
            "Faith": ("A_Faith", "B_Faith", False),
            "Rel": ("A_Rel", "B_Rel", False),
            "ROUGE": ("A_ROUGE_L", "B_ROUGE_L", False),
        }

        results_a = {}
        results_b = {}

        for name, (col_a, col_b, is_pct) in metrics.items():
            val_a = df[col_a].mean()
            val_b = df[col_b].mean()

            if is_pct:
                results_a[name] = f"{val_a * 100:.1f}%"
                results_b[name] = f"{val_b * 100:.1f}%"
            else:
                precision = 4 if name == "ROUGE" else 2
                results_a[name] = f"{val_a:.{precision}f}"
                results_b[name] = f"{val_b:.{precision}f}"

        num_questions = len(df)

        print(f"\n==================== 评测汇总（共 {num_questions} 题） ====================")
        print("指标                         | Method A (固定切分) | Method B (BBox切分)")
        print("-" * 76)
        print(f"Hit@5                       | {results_a['Hit@5']:<18} | {results_b['Hit@5']:<18}")
        print(f"MRR@5                       | {results_a['MRR@5']:<18} | {results_b['MRR@5']:<18}")
        print(f"Context Precision           | {results_a['CPrec']:<18} | {results_b['CPrec']:<18}")
        print(f"Faithfulness (0-10)         | {results_a['Faith']:<18} | {results_b['Faith']:<18}")
        print(f"Relevance (0-10)            | {results_a['Rel']:<18} | {results_b['Rel']:<18}")
        print(f"ROUGE-L                     | {results_a['ROUGE']:<18} | {results_b['ROUGE']:<18}")
        print("-" * 76)

    except Exception as e:
        print(f"读取 CSV 或生成报告失败: {e}")


if __name__ == "__main__":
    generate_summary_from_csv("record/eval_Qwen3.5-2B_chunk200.csv")
