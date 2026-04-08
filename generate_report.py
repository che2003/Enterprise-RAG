import pandas as pd
import numpy as np


def generate_summary_from_csv(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)

        # 提取核心指标
        metrics = {
            "Hit@5": ("A_Hit", "B_Hit", True),  # True表示需转为百分比
            "MRR@5": ("A_MRR", "B_MRR", False),
            "CPrec": ("A_CPrec", "B_CPrec", False),
            "Faith": ("A_Faith", "B_Faith", False),
            "Rel": ("A_Rel", "B_Rel", False),
            "ROUGE": ("A_ROUGE_L", "B_ROUGE_L", False)
        }

        results_A = {}
        results_B = {}

        for name, (col_A, col_B, is_pct) in metrics.items():
            val_A = df[col_A].mean()
            val_B = df[col_B].mean()

            if is_pct:
                results_A[name] = f"{val_A * 100:.1f}      %"
                results_B[name] = f"{val_B * 100:.1f}      %"
            else:
                results_A[name] = f"{val_A:.4f}      " if name == "ROUGE" else f"{val_A:.2f}      "
                results_B[name] = f"{val_B:.4f}      " if name == "ROUGE" else f"{val_B:.2f}      "

        num_questions = len(df)

        print(f"\n==================== 📊 终极大盘 (基于 CSV 读取, 共 {num_questions} 题) ====================")
        print("评测维度 (核心指标)            | Method A: 固定切分     | Method B: BBox 降噪 ")
        print("-" * 65)
        print(f"1. 有效召回率 (Hit@5)     | {results_A['Hit@5']:<18} | {results_B['Hit@5']:<18}")
        print(f"2. 平均倒数排名 (MRR@5)    | {results_A['MRR@5']:<18} | {results_B['MRR@5']:<18}")
        print(f"3. 上下文纯净度 (CPrec)    | {results_A['CPrec']:<18} | {results_B['CPrec']:<18}")
        print(f"4. 答案忠实度 (0-10)      | {results_A['Faith']:<18} | {results_B['Faith']:<18}")
        print(f"5. 答案相关性 (0-10)      | {results_A['Rel']:<18} | {results_B['Rel']:<18}")
        print(f"6. 专家对齐度 (ROUGE-L)   | {results_A['ROUGE']:<18} | {results_B['ROUGE']:<18}")
        print("-" * 65)

    except Exception as e:
        print(f"❌ 读取 CSV 或生成报告时出错: {e}")


# 兼容直接单独运行此脚本的情况 (填入你最后一次生成的CSV名字用于测试)
if __name__ == "__main__":
    # 如果你想单独运行这个脚本看结果，可以把下面的路径改成你想看的CSV文件
    generate_summary_from_csv("record/eval_Qwen3.5-2B_chunk200.csv")
    # pass