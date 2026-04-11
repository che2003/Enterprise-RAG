# Enterprise-RAG

一个面向论文问答场景的 RAG（Retrieval-Augmented Generation）实验项目，支持：

- 双路检索（Dense + Sparse）与 RRF 融合。
- 两种切分策略（固定切分 / BBox 版面切分）对比。
- 基于 QASper 数据集的自动化评测与断点续跑。
- Gradio WebUI 的知识库构建、问答与评测看板。

## 1. 项目结构

```text
Enterprise-RAG/
├── app.py                    # Gradio WebUI
├── main.py                   # 命令行批量评测入口
├── generate_report.py        # CSV 汇总报告生成
├── run_experiments.sh        # 一键跑实验矩阵
├── requirements.txt
├── data/
│   ├── demo_chunks.json      # Demo 索引数据
│   └── raw_papers/           # 下载的论文 PDF（运行时生成）
├── record/                   # 评测输出（运行时生成）
└── src/
    ├── data_pipeline.py      # 数据拉取与文档切分
    ├── hybrid_retriever.py   # 检索与索引
    └── llm_evaluator.py      # 生成与裁判评估
```

## 2. 功能说明

### 2.1 数据与切分

- `DataPipeline.fetch_qasper_sample()`：从 Hugging Face 的 QASper 验证集拉取样本，并下载对应 arXiv PDF。
- `naive_fixed_chunking()`：固定窗口切分。
- `bbox_layout_chunking()`：基于版面 block 的顺序切分（Full-width/Left/Right）。

### 2.2 检索

`HybridRetriever` 同时构建：

- FAISS 向量索引（Dense）。
- BM25 倒排检索（Sparse）。

检索时支持三种模式：

- `dense`：仅向量检索。
- `sparse`：仅 BM25。
- `hybrid`：RRF 融合排序。

并支持 `target_doc_name` 做文档级过滤，避免跨文档串检索。

### 2.3 评测

`main.py` 输出两类结果：

1. 检索指标：`Hit@5`、`MRR@5`、`Context Precision`。
2. 生成质量指标：`Faithfulness`、`Relevance`、`ROUGE-L`。

结果写入：

- `record/eval_*.csv`：结构化结果，支持断点续跑。
- `record/details_*.txt`：逐题详细结果。

### 2.4 WebUI

`app.py` 提供：

- PDF 上传后构建 Method A / Method B 索引。
- 基于检索策略与文档过滤的问答。
- 自动读取最新 CSV 并展示评测看板。
- Demo 数据一键加载。

## 3. 环境准备

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

> 说明：如果你在离线环境使用 embedding / LLM，请先准备本地模型目录。

## 4. 配置项（环境变量）

### 检索相关

- `RAG_EMBED_MODEL_PATH`：本地 embedding 模型路径（优先使用）。
- `RAG_ALLOW_ONLINE_MODEL=1`：离线加载失败后允许在线下载。
- `RAG_METHOD_B_MAX_CHUNKS`：Method B 最大 chunks 上限（默认 1500）。

### LLM 相关

- `RAG_GENERATOR_ID`：生成模型路径或模型名。
- `RAG_JUDGE_ID`：裁判模型路径或模型名。
- `RAG_LOCAL_FILES_ONLY=1`：仅从本地加载模型。

## 5. 使用方式

### 5.1 命令行批量评测

```bash
python main.py \
  --num_papers 10 \
  --chunk_size 400 \
  --gen_model Qwen3.5-2B \
  --judge_model Qwen3.5-9B \
  --retrieval_mode hybrid
```

### 5.2 一键跑实验矩阵

```bash
bash run_experiments.sh
```

### 5.3 启动 WebUI

```bash
python app.py
```

启动后在终端输出的本地地址打开页面（通常为 `http://127.0.0.1:7860`）。


### 5.4 生成论文图（图2/图3/图4）

当 `record/` 下已有对应实验 CSV 后，可直接生成三张图：

```bash
python plot_paper_figures.py
```

可选参数示例：

```bash
python plot_paper_figures.py \
  --record_dir record \
  --output_dir record/figures \
  --line_chunk_sizes 200 400 600 800 \
  --ablation_chunk_size 400
```

## 6. 输出结果解读

### CSV 核心列

- `A_*`：Method A（固定切分）指标。
- `B_*`：Method B（BBox 切分）指标。
- `A_Hit/B_Hit`：是否命中相关 chunk。
- `A_MRR/B_MRR`：首个相关 chunk 排名倒数。
- `A_CPrec/B_CPrec`：Top-K 上下文纯度。
- `A_Faith/B_Faith`：答案忠实度（裁判模型打分）。
- `A_Rel/B_Rel`：答案相关度（裁判模型打分）。
- `A_ROUGE_L/B_ROUGE_L`：与参考答案重合程度。

### 自动报告

```bash
python generate_report.py
```

或在 `main.py` 评测结束后自动触发汇总。

## 7. 常见问题

1. **模型加载失败**
   - 先确认模型路径是否可读。
   - 离线场景建议设置本地路径，并保持 `RAG_LOCAL_FILES_ONLY=1`。

2. **构建索引很慢**
   - 适当减小 `--num_papers` 或 `--chunk_size`。
   - 限制 `RAG_METHOD_B_MAX_CHUNKS`，避免 Method B 生成过多 chunk。

3. **CSV 不完整 / 中断后续跑**
   - 本项目按 `Question` 去重并支持断点续跑。
   - 相同参数再次运行会自动跳过已完成题目。

## 8. 许可证与用途

本仓库主要用于学习与研究性质的 RAG 实验验证，请根据你所在组织和数据来源要求合规使用。
