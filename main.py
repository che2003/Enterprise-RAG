import argparse
import csv
import datetime
import os
import string

from generate_report import generate_summary_from_csv
from src.data_pipeline import DataPipeline
from src.hybrid_retriever import HybridRetriever
from src.llm_evaluator import RAGEvaluator


def evaluate_retrieval_metrics(retrieved_chunks_list, query):
    stop_words = {
        "what", "is", "the", "in", "of", "to", "a", "an", "and", "for", "are",
        "do", "they", "with", "how", "did", "use", "on", "from",
    }
    query_terms = [
        w.strip(string.punctuation).lower()
        for w in query.split()
        if w.lower() not in stop_words and len(w) > 2
    ]

    if not query_terms:
        return 1.0, 1.0, 1.0

    threshold = 1 if len(query_terms) <= 2 else 2
    relevance_array = []
    for chunk in retrieved_chunks_list:
        chunk_lower = chunk.lower()
        match_count = sum(1 for term in query_terms if term in chunk_lower)
        relevance_array.append(1 if match_count >= threshold else 0)

    hit = 1 if sum(relevance_array) > 0 else 0
    mrr = 0.0
    for idx, rel in enumerate(relevance_array):
        if rel == 1:
            mrr = 1.0 / (idx + 1)
            break
    context_precision = sum(relevance_array) / len(relevance_array) if relevance_array else 0.0

    return hit, mrr, context_precision


def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG 批量评测")
    parser.add_argument("--num_papers", type=int, default=10, help="拉取 QASper 论文数量")
    parser.add_argument("--chunk_size", type=int, default=400, help="切分块大小")
    parser.add_argument("--gen_model", type=str, default="Qwen3.5-2B", help="生成模型")
    parser.add_argument("--judge_model", type=str, default="Qwen3.5-9B", help="评估模型")
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "dense", "sparse"],
        help="检索模式",
    )
    args = parser.parse_args()

    print("\n==================== Enterprise RAG Evaluation ====================")

    record_dir = "record"
    os.makedirs(record_dir, exist_ok=True)

    num_papers = args.num_papers
    chunk_size = args.chunk_size
    gen_model = args.gen_model
    judge_model = args.judge_model
    retrieval_mode = args.retrieval_mode

    safe_gen_name = gen_model.replace("/", "-")
    csv_filename = f"eval_{safe_gen_name}_chunk{chunk_size}_mode-{retrieval_mode}.csv"
    csv_filepath = os.path.join(record_dir, csv_filename)
    txt_filepath = os.path.join(record_dir, f"details_{safe_gen_name}_chunk{chunk_size}_mode-{retrieval_mode}.txt")

    processed_questions = set()
    if os.path.exists(csv_filepath):
        with open(csv_filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_questions.add(row["Question"])
        print(f"检测到历史评测记录，已恢复 {len(processed_questions)} 道题目。")
    else:
        with open(csv_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Parameters", "Q_ID", "Question", "Ground_Truth",
                "A_Hit", "A_MRR", "A_CPrec", "A_Faith", "A_Rel", "A_ROUGE_L",
                "B_Hit", "B_MRR", "B_CPrec", "B_Faith", "B_Rel", "B_ROUGE_L",
            ])

    pipeline = DataPipeline()
    pdf_paths, eval_qas = pipeline.fetch_qasper_sample(num_papers=num_papers)
    if not pdf_paths:
        return

    print(f"\n构建 Method A 索引（固定切分，chunk size={chunk_size}）...")
    chunks_a = pipeline.naive_fixed_chunking(pdf_paths, chunk_size=chunk_size, overlap=50)
    retriever_a = HybridRetriever()
    retriever_a.build_index(chunks_a)

    print(f"构建 Method B 索引（BBox 切分，chunk size={chunk_size}）...")
    chunks_b = pipeline.bbox_layout_chunking(pdf_paths, target_chunk_size=chunk_size)
    retriever_b = HybridRetriever()
    retriever_b.build_index(chunks_b)

    evaluator = RAGEvaluator(generator_id=rf"D:\models\{gen_model}", judge_id=rf"D:\models\{judge_model}")
    num_q = len(eval_qas)

    print(f"\n开始评测，共 {num_q} 道题目。")

    with open(csv_filepath, "a", newline="", encoding="utf-8") as f_csv, open(txt_filepath, "a", encoding="utf-8") as f_txt:
        csv_writer = csv.writer(f_csv)

        for i, (query, ground_truth, doc_name) in enumerate(eval_qas):
            if query in processed_questions:
                continue

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            param_str = f"Chunk:{chunk_size}|Gen:{gen_model}|Judge:{judge_model}|Mode:{retrieval_mode}"

            print(f"\n[Q {i + 1}/{num_q}] (Doc: {doc_name}) {query}")
            f_txt.write(f"\n{'=' * 40}\n[Q {i + 1}/{num_q}] {query}\n[Ground Truth]: {ground_truth}\n{'=' * 40}\n")

            context_a_str, chunks_a_list = retriever_a.hybrid_search_rrf(
                query,
                top_k=5,
                target_doc_name=doc_name,
                mode=retrieval_mode,
            )
            hit_a, mrr_a, cprec_a = evaluate_retrieval_metrics(chunks_a_list, query)
            ans_a = evaluator.generate_answer(query, context_a_str)
            score_a = evaluator.evaluate_as_judge(query, context_a_str, ans_a, ground_truth)
            rouge_a = evaluator.compute_rouge_l(ans_a, ground_truth)

            f_txt.write(f"\n[Method A]\n[Answer]:\n{ans_a}\n")
            f_txt.write(f"[IR]: Hit={hit_a} | MRR={mrr_a:.2f} | CPrec={cprec_a:.2f}\n")
            f_txt.write(
                f"[Judge]: Faithfulness={score_a['Faithfulness']} | Relevance={score_a['Relevance']} | ROUGE-L={rouge_a:.4f}\n"
            )

            context_b_str, chunks_b_list = retriever_b.hybrid_search_rrf(
                query,
                top_k=5,
                target_doc_name=doc_name,
                mode=retrieval_mode,
            )
            hit_b, mrr_b, cprec_b = evaluate_retrieval_metrics(chunks_b_list, query)
            ans_b = evaluator.generate_answer(query, context_b_str)
            score_b = evaluator.evaluate_as_judge(query, context_b_str, ans_b, ground_truth)
            rouge_b = evaluator.compute_rouge_l(ans_b, ground_truth)

            f_txt.write(f"\n[Method B]\n[Answer]:\n{ans_b}\n")
            f_txt.write(f"[IR]: Hit={hit_b} | MRR={mrr_b:.2f} | CPrec={cprec_b:.2f}\n")
            f_txt.write(
                f"[Judge]: Faithfulness={score_b['Faithfulness']} | Relevance={score_b['Relevance']} | ROUGE-L={rouge_b:.4f}\n"
            )

            print(
                f"A: MRR={mrr_a:.2f} | CPrec={cprec_a:.2f} | Faith={score_a['Faithfulness']:>2} | "
                f"Rel={score_a['Relevance']:>2} | ROUGE={rouge_a:.2f}"
            )
            print(
                f"B: MRR={mrr_b:.2f} | CPrec={cprec_b:.2f} | Faith={score_b['Faithfulness']:>2} | "
                f"Rel={score_b['Relevance']:>2} | ROUGE={rouge_b:.2f}"
            )

            csv_writer.writerow([
                current_time, param_str, i + 1, query, ground_truth,
                hit_a, mrr_a, cprec_a, score_a["Faithfulness"], score_a["Relevance"], rouge_a,
                hit_b, mrr_b, cprec_b, score_b["Faithfulness"], score_b["Relevance"], rouge_b,
            ])
            f_csv.flush()
            f_txt.flush()
            os.fsync(f_csv.fileno())
            os.fsync(f_txt.fileno())
            processed_questions.add(query)

    print(f"\n评测完成，结果写入 {csv_filepath}")
    try:
        print("\n==================== 自动汇总报告 ====================")
        generate_summary_from_csv(csv_filepath)
    except Exception as e:
        print(f"自动生成报告失败: {e}")


if __name__ == "__main__":
    main()
