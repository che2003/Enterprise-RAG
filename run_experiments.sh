#!/bin/bash

# 全局参数：为了报告的数据说服力，将样本数扩大
NUM_PAPERS=30
GEN_MODEL="Qwen3.5-2B"
JUDGE_MODEL="Qwen3.5-9B"

echo "==============================================================="
echo "🚀 启动企业级 RAG 自动化压测矩阵"
echo "本次评测文档数: $NUM_PAPERS"
echo "==============================================================="

# -------------------------------------------------------------
# 实验一：Chunk Size 敏感性分析 (固定使用混合检索)
# 目的：证明 Method B (BBox 降噪) 比 Method A 对 Chunk Size 容错率更高
# -------------------------------------------------------------
echo -e "\n[实验一] 开始跑测 Chunk Size 消融 (200, 400, 600, 800)..."

for CHUNK in 200 400 600 800
do
    echo "▶️ 当前运行参数 -> Chunk Size: $CHUNK | Mode: hybrid"
    python main.py \
        --num_papers "$NUM_PAPERS" \
        --chunk_size "$CHUNK" \
        --gen_model "$GEN_MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --retrieval_mode hybrid
done

# -------------------------------------------------------------
# 实验二：检索策略消融实验 (固定 Chunk Size 为 400)
# 目的：证明 FAISS(稠密) + BM25(稀疏) 双路 RRF 融合效果 > 单路效果
# -------------------------------------------------------------
echo -e "\n[实验二] 开始跑测检索策略消融 (dense, sparse)..."

# hybrid 模式已经在上面的实验一跑过 400 的了，这里只跑 dense 和 sparse
for MODE in dense sparse
do
    echo "▶️ 当前运行参数 -> Chunk Size: 400 | Mode: $MODE"
    python main.py \
        --num_papers "$NUM_PAPERS" \
        --chunk_size 400 \
        --gen_model "$GEN_MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --retrieval_mode "$MODE"
done

echo "==============================================================="
echo "🎉 所有实验矩阵执行完毕！请查看 record/ 目录获取所有 CSV 报告。"
echo "==============================================================="
