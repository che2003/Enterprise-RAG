import os
import csv
import string
import datetime
from src.data_pipeline import DataPipeline
from src.hybrid_retriever import HybridRetriever
from src.llm_evaluator import RAGEvaluator
from generate_report import generate_summary_from_csv


def evaluate_retrieval_metrics(retrieved_chunks_list, query):
    stop_words = {"what", "is", "the", "in", "of", "to", "a", "an", "and", "for", "are", "do", "they", "with", "how",
                  "did", "use", "on", "from"}
    query_terms = [w.strip(string.punctuation).lower() for w in query.split() if
                   w.lower() not in stop_words and len(w) > 2]

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
    print(f"\n==================== 🚀 DSAI5201 终极压测 (异构裁判解耦 + 断点续跑) ====================")

    record_dir = "record"
    os.makedirs(record_dir, exist_ok=True)

    # ==========================================
    # 💡 核心超参数区
    # ==========================================
    NUM_PAPERS = 10  # 论文拉取数量
    PARAM_CHUNK_SIZE = 300  # 文本切分大小
    PARAM_GEN_MODEL = "Qwen3.5-2B"  # 选手模型
    PARAM_JUDGE_MODEL = "Qwen3.5-9B"  # 裁判模型

    safe_gen_name = PARAM_GEN_MODEL.replace("/", "-")
    csv_filename = f"eval_{safe_gen_name}_chunk{PARAM_CHUNK_SIZE}.csv"
    csv_filepath = os.path.join(record_dir, csv_filename)
    txt_filepath = os.path.join(record_dir, f"details_{safe_gen_name}_chunk{PARAM_CHUNK_SIZE}.txt")

    processed_questions = set()
    if os.path.exists(csv_filepath):
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader: processed_questions.add(row['Question'])
        print(f"🔄 检测到【同参数】历史测试记录！已成功恢复 {len(processed_questions)} 道已测题目...")
    else:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Parameters", "Q_ID", "Question", "Ground_Truth",
                "A_Hit", "A_MRR", "A_CPrec", "A_Faith", "A_Rel", "A_ROUGE_L",
                "B_Hit", "B_MRR", "B_CPrec", "B_Faith", "B_Rel", "B_ROUGE_L"
            ])

    pipeline = DataPipeline()
    pdf_paths, eval_qas = pipeline.fetch_qasper_sample(num_papers=NUM_PAPERS)
    if not pdf_paths: return

    print(f"\n>> 构建 Method A (纯文本固定字数切分, Chunk Size: {PARAM_CHUNK_SIZE})...")
    chunks_A = pipeline.naive_fixed_chunking(pdf_paths, chunk_size=PARAM_CHUNK_SIZE, overlap=50)
    retriever_A = HybridRetriever()
    retriever_A.build_index(chunks_A)

    print(f">> 构建 Method B (物理 BBox + 柔性过滤 + 细粒度聚合, Chunk Size: {PARAM_CHUNK_SIZE})...")
    chunks_B = pipeline.bbox_layout_chunking(pdf_paths, target_chunk_size=PARAM_CHUNK_SIZE)
    retriever_B = HybridRetriever()
    retriever_B.build_index(chunks_B)

    # 唤醒异构引擎
    evaluator = RAGEvaluator(
        generator_id=rf"D:\models\{PARAM_GEN_MODEL}",
        judge_id=rf"D:\models\{PARAM_JUDGE_MODEL}"
    )
    num_q = len(eval_qas)

    print(f"\n==================== 开始异构自动化评测 (共 {num_q} 题) ====================")

    with open(csv_filepath, 'a', newline='', encoding='utf-8') as f_csv, open(txt_filepath, 'a',
                                                                              encoding='utf-8') as f_txt:
        csv_writer = csv.writer(f_csv)

        # 💡 解包时接收三个参数：新增 doc_name！
        for i, (query, ground_truth, doc_name) in enumerate(eval_qas):
            if query in processed_questions: continue

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            param_str = f"Chunk:{PARAM_CHUNK_SIZE}|Gen:{PARAM_GEN_MODEL}"

            # 打印日志带上论文名，方便溯源
            print(f"\n❓ [Q {i + 1}/{num_q}] (Doc: {doc_name}): {query}")
            f_txt.write(f"\n{'=' * 40}\n❓ [Q {i + 1}/{num_q}]: {query}\n[Ground Truth]: {ground_truth}\n{'=' * 40}\n")

            # ---------------- Method A ----------------
            # 💡 传入 target_doc_name 触发硬拦截，彻底告别串库
            context_A_str, chunks_A_list = retriever_A.hybrid_search_rrf(query, top_k=5, target_doc_name=doc_name)
            hit_A, mrr_A, cprec_A = evaluate_retrieval_metrics(chunks_A_list, query)
            ans_A = evaluator.generate_answer(query, context_A_str)
            score_A = evaluator.evaluate_as_judge(query, context_A_str, ans_A, ground_truth)
            rouge_A = evaluator.compute_rouge_l(ans_A, ground_truth)

            f_txt.write(f"\n🔴 [Method A]\n【生成答案】:\n{ans_A}\n")
            f_txt.write(f"【底层 IR 指标】: Hit={hit_A} | MRR={mrr_A:.2f} | Context Prec={cprec_A:.2f}\n")
            f_txt.write(
                f"【9B 裁判打分】: 忠实度={score_A['Faithfulness']} | 相关性={score_A['Relevance']} | ROUGE-L={rouge_A:.4f}\n")

            # ---------------- Method B ----------------
            # 💡 同样传入 target_doc_name 防串库
            context_B_str, chunks_B_list = retriever_B.hybrid_search_rrf(query, top_k=5, target_doc_name=doc_name)
            hit_B, mrr_B, cprec_B = evaluate_retrieval_metrics(chunks_B_list, query)
            ans_B = evaluator.generate_answer(query, context_B_str)
            score_B = evaluator.evaluate_as_judge(query, context_B_str, ans_B, ground_truth)
            rouge_B = evaluator.compute_rouge_l(ans_B, ground_truth)

            f_txt.write(f"\n🟢 [Method B]\n【生成答案】:\n{ans_B}\n")
            f_txt.write(f"【底层 IR 指标】: Hit={hit_B} | MRR={mrr_B:.2f} | Context Prec={cprec_B:.2f}\n")
            f_txt.write(
                f"【9B 裁判打分】: 忠实度={score_B['Faithfulness']} | 相关性={score_B['Relevance']} | ROUGE-L={rouge_B:.4f}\n")

            print(
                f"🔴 [A] MRR: {mrr_A:.2f} | 纯净度: {cprec_A:.2f} | 忠实: {score_A['Faithfulness']:>2} | 相关: {score_A['Relevance']:>2} | ROUGE: {rouge_A:.2f}")
            print(
                f"🟢 [B] MRR: {mrr_B:.2f} | 纯净度: {cprec_B:.2f} | 忠实: {score_B['Faithfulness']:>2} | 相关: {score_B['Relevance']:>2} | ROUGE: {rouge_B:.2f}")

            csv_writer.writerow([
                current_time, param_str, i + 1, query, ground_truth,
                hit_A, mrr_A, cprec_A, score_A['Faithfulness'], score_A['Relevance'], rouge_A,
                hit_B, mrr_B, cprec_B, score_B['Faithfulness'], score_B['Relevance'], rouge_B
            ])
            f_csv.flush()
            f_txt.flush()
            os.fsync(f_csv.fileno())
            os.fsync(f_txt.fileno())

            processed_questions.add(query)

    print(f"\n🎉 双模型异构压测大满贯完成！详细对标数据已写入 {record_dir} 文件夹的 {csv_filename} 中。")
    try:
        print("\n" + "=" * 20 + " 📊 自动拉取大盘分析报告 " + "=" * 20)
        generate_summary_from_csv(csv_filepath)
    except Exception as e:
        print(f"⚠️ 自动生成报告时出错，请检查逻辑或路径: {e}")


if __name__ == "__main__":
    main()