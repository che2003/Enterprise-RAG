#!/bin/bash

NUM_PAPERS=30
GEN_MODEL="Qwen3.5-2B"
JUDGE_MODEL="Qwen3.5-9B"

echo "==============================================================="
echo "启动 RAG 自动化评测"
echo "评测文档数: $NUM_PAPERS"
echo "==============================================================="

echo -e "\n[实验一] Chunk Size 消融: 200, 400, 600, 800"
for CHUNK in 200 400 600 800
do
    echo "运行参数: chunk_size=$CHUNK, retrieval_mode=hybrid"
    python main.py \
        --num_papers "$NUM_PAPERS" \
        --chunk_size "$CHUNK" \
        --gen_model "$GEN_MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --retrieval_mode hybrid
done

echo -e "\n[实验二] 检索模式消融: dense, sparse"
for MODE in dense sparse
do
    echo "运行参数: chunk_size=400, retrieval_mode=$MODE"
    python main.py \
        --num_papers "$NUM_PAPERS" \
        --chunk_size 400 \
        --gen_model "$GEN_MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --retrieval_mode "$MODE"
done

echo "==============================================================="
echo "全部实验执行完成，请查看 record/ 目录。"
echo "==============================================================="
