import glob
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import gradio as gr
import pandas as pd
from src.data_pipeline import DataPipeline
from src.hybrid_retriever import HybridRetriever
from src.llm_evaluator import RAGEvaluator
print("==================================================")
print("🚀 正在启动企业级 RAG WebUI 服务，请稍候...")
print("==================================================")
# ==========================================
# 全局引擎与系统状态
# ==========================================
pipeline: Optional[DataPipeline] = None
retriever_A: Optional[HybridRetriever] = None
retriever_B: Optional[HybridRetriever] = None
evaluator: Optional[RAGEvaluator] = None
model_init_attempted = False
SYSTEM_STATE: Dict[str, object] = {
    "embedding_ready": False,
    "model_ready": False,
    "engine_status": "冷启动未完成（首次使用时按需初始化）",
    "index_ready": False,
    "index_summary": "尚未构建索引",
    "dashboard_summary": "尚未读取评测 CSV",
}
DEMO_CHUNKS_PATH = os.path.join("data", "demo_chunks.json")
def resolve_method_b_chunk_limit() -> int:
    raw = os.getenv("RAG_METHOD_B_MAX_CHUNKS", "1500").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1500
    return max(100, value)
def ts() -> str:
    return time.strftime("%H:%M:%S")
def log_event(message: str) -> None:
    print(f"[{ts()}] {message}")
def build_system_status_markdown() -> str:
    embedding_flag = "✅" if SYSTEM_STATE["embedding_ready"] else "❌"
    model_flag = "✅" if SYSTEM_STATE["model_ready"] else "❌"
    index_flag = "✅" if SYSTEM_STATE["index_ready"] else "⚠️"
    return (
        "### 🟢 系统统一状态栏\n"
        f"- Embedding 引擎：{embedding_flag}\n"
        f"- LLM 生成引擎：{model_flag}\n"
        f"- 冷启动状态：{SYSTEM_STATE['engine_status']}\n"
        f"- 检索索引：{index_flag}\n"
        f"- 索引摘要：{SYSTEM_STATE['index_summary']}\n"
        f"- 大盘读取状态：{SYSTEM_STATE['dashboard_summary']}"
    )
def success_error_payload(error_code: str, message: str, action: str) -> str:
    return json.dumps(
        {"error_code": error_code, "message": message, "action": action},
        ensure_ascii=False,
        indent=2,
    )
def ensure_retrieval_engines() -> Tuple[bool, str]:
    global pipeline, retriever_A, retriever_B
    if pipeline is not None and retriever_A is not None and retriever_B is not None:
        log_event("检索引擎复用已初始化实例。")
        return True, ""
    SYSTEM_STATE["engine_status"] = "检索引擎冷启动中..."
    log_event("开始冷启动检索引擎（DataPipeline + 双路 HybridRetriever）。")
    try:
        pipeline = DataPipeline()
        retriever_A = HybridRetriever()
        retriever_B = HybridRetriever()
        SYSTEM_STATE["embedding_ready"] = True
        SYSTEM_STATE["engine_status"] = "检索引擎已就绪"
        log_event("检索引擎冷启动完成。")
        return True, ""
    except Exception as e:
        SYSTEM_STATE["embedding_ready"] = False
        SYSTEM_STATE["engine_status"] = f"检索引擎初始化失败: {e}"
        log_event(f"检索引擎冷启动失败: {e}")
        return False, str(e)
def ensure_evaluator_engine() -> Tuple[bool, str]:
    global evaluator, model_init_attempted
    if evaluator is not None:
        log_event("LLM 引擎复用已加载模型。")
        return True, ""
    if model_init_attempted:
        log_event("LLM 初始化曾失败，跳过重复初始化。")
        return False, "模型初始化已失败，请检查模型路径/网络或显存"
    model_init_attempted = True
    SYSTEM_STATE["engine_status"] = "LLM 引擎冷启动中..."
    log_event("开始冷启动 LLM 引擎。")
    try:
        generator_id = os.getenv("RAG_GENERATOR_ID", "Qwen/Qwen3.5-2B")
        judge_id = os.getenv("RAG_JUDGE_ID", r"D:\models\Qwen3.5-9B")
        local_files_only = os.getenv("RAG_LOCAL_FILES_ONLY", "1") == "1"
        log_event(
            f"LLM 配置: generator={generator_id}, judge={judge_id}, local_files_only={local_files_only}"
        )
        evaluator = RAGEvaluator(
            generator_id=generator_id,
            judge_id=judge_id,
            local_files_only=local_files_only,
        )
        SYSTEM_STATE["model_ready"] = True
        SYSTEM_STATE["engine_status"] = "LLM 引擎已就绪"
        log_event("LLM 引擎冷启动完成。")
        return True, ""
    except Exception as e:
        SYSTEM_STATE["model_ready"] = False
        SYSTEM_STATE["engine_status"] = f"LLM 初始化失败: {e}"
        log_event(f"LLM 引擎冷启动失败: {e}")
        return False, str(e)
def chat_with_rag(
    user_message: str,
    history: List[Dict[str, str]],
    strategy: str,
    top_k: int,
    target_doc: str,
):
    log_event(f"收到问答请求: strategy={strategy}, top_k={top_k}, target_doc={target_doc}")
    if not user_message.strip():
        log_event("问答请求被拒绝：空问题。")
        return "", history, "请输入有效问题。", "{}", build_system_status_markdown()
    time.sleep(0.2)
    doc_filter = target_doc if target_doc != "🌍 全局检索 (混合所有文档)" else None
    ready, init_error = ensure_retrieval_engines()
    if not ready:
        log_event(f"问答失败：检索引擎未就绪。原因: {init_error}")
        error_payload = success_error_payload(
            "ENGINE_INIT_FAILED", init_error, "请检查依赖与日志后重启服务"
        )
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "❌ 检索引擎初始化失败"},
            ]
        )
        return "", history, "引擎未就绪，无法提供溯源。", error_payload, build_system_status_markdown()
    try:
        retriever = retriever_A if "Method A" in strategy else retriever_B
        context_str, chunks_list = retriever.hybrid_search_rrf(
            user_message,
            top_k=int(top_k),
            target_doc_name=doc_filter,
        )
        log_event(f"检索完成: 命中 chunks={len(chunks_list)}")
    except ValueError as e:
        log_event(f"问答失败：索引未就绪。原因: {e}")
        error_payload = success_error_payload(
            "INDEX_NOT_READY", str(e), "请先上传 PDF 建库，或点击「先看看效果（加载 Demo 数据）」"
        )
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "⚠️ 当前未建索引，请先建库。"},
            ]
        )
        return "", history, "暂无溯源数据。", error_payload, build_system_status_markdown()
    except Exception as e:
        log_event(f"问答失败：检索阶段异常。原因: {e}")
        error_payload = success_error_payload(
            "RAG_RUNTIME_ERROR",
            str(e),
            "请重试；如仍失败，请重新构建索引并检查模型加载状态",
        )
        history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "❌ 检索过程中发生系统异常。"},
            ]
        )
        return "", history, "系统异常，溯源不可用。", error_payload, build_system_status_markdown()
    if not chunks_list:
        log_event("问答进入兜底回复：未命中任何 chunk。")
        answer = "知识库中暂未检索到相关内容，请尝试更换问题或扩充知识库。"
        context_str = "未命中任何文档片段。"
        error_payload = "{}"
    else:
        model_ready, _ = ensure_evaluator_engine()
        if model_ready and evaluator is not None:
            answer = evaluator.generate_answer(user_message, context_str)
            log_event("问答生成完成：LLM 已返回答案。")
            error_payload = "{}"
        else:
            log_event("问答降级：LLM 未就绪，返回检索摘要模式。")
            answer = "模型冷启动失败，当前返回检索摘要模式。请检查模型路径或显存。"
            error_payload = success_error_payload(
                "MODEL_NOT_READY",
                "生成模型未就绪",
                "请检查模型路径、显存容量并重启服务",
            )
    history.extend(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ]
    )
    log_event(f"问答结束：history_size={len(history)}")
    return "", history, context_str, error_payload, build_system_status_markdown()
def build_knowledge_base(file_objs, chunk_size):
    if not file_objs:
        yield (
            "⚠️ 未检测到文件，请先上传 PDF。",
            gr.update(),
            "⚠️ 索引未就绪",
            build_system_status_markdown(),
            "{}"
        )
        return
    try:
        chunk_size = int(chunk_size)
    except Exception:
        yield (
            "⚠️ chunk_size 非法，请输入有效整数（例如 400）。",
            gr.update(),
            "⚠️ 索引未就绪",
            build_system_status_markdown(),
            success_error_payload("INVALID_CHUNK_SIZE", "chunk_size 非法", "请输入整数并重试")
        )
        return
    ready, init_error = ensure_retrieval_engines()
    if not ready:
        yield (
            f"❌ 检索引擎初始化失败：{init_error}",
            gr.update(),
            "❌ 索引构建失败",
            build_system_status_markdown(),
            success_error_payload("ENGINE_INIT_FAILED", init_error, "优先使用本地 embedding 模型（RAG_EMBED_MODEL_PATH）后重试")
        )
        return
    start = time.time()
    logs = []
    file_paths = [f.name for f in file_objs]
    doc_names = [os.path.basename(p) for p in file_paths]
    dropdown_choices = ["🌍 全局检索 (混合所有文档)"] + doc_names
    logs.append(f"[{ts()}] 📦 收到 {len(file_paths)} 篇文献: {', '.join(doc_names)}")
    logs.append(f"[{ts()}] ⚙️ 当前切分块大小设置: {chunk_size}")
    log_event(f"开始建库：docs={len(file_paths)}, chunk_size={chunk_size}")
    yield "\n".join(logs), gr.update(choices=dropdown_choices, value=dropdown_choices[0]), "⏳ 索引构建中...", build_system_status_markdown(), "{}"
    try:
        logs.append(f"\n[{ts()}] >> 正在构建 Method A (基础固定切分)...")
        yield "\n".join(logs), gr.update(), "⏳ 索引构建中...", build_system_status_markdown(), "{}"
        chunks_A = pipeline.naive_fixed_chunking(file_paths, chunk_size=chunk_size, overlap=50)
        logs.append(f"[{ts()}] Method A 切分完成，准备写入向量/BM25 索引...")
        retriever_A.build_index(chunks_A)
        logs.append(f"[{ts()}] ✅ Method A 构建完成，共生成 {len(chunks_A)} 个 Chunk。")
        yield "\n".join(logs), gr.update(), "⏳ 索引构建中...", build_system_status_markdown(), "{}"
        logs.append(f"\n[{ts()}] >> 正在构建 Method B (物理 BBox 降噪切分)...")
        yield "\n".join(logs), gr.update(), "⏳ 索引构建中...", build_system_status_markdown(), "{}"
        chunks_B = pipeline.bbox_layout_chunking(file_paths, target_chunk_size=chunk_size)
        max_b_chunks = resolve_method_b_chunk_limit()
        if len(chunks_B) > max_b_chunks:
            logs.append(
                f"[{ts()}] ⚠️ Method B chunk 数过大（{len(chunks_B)}），已截断到 {max_b_chunks} 以避免长时间无响应。"
            )
            chunks_B = chunks_B[:max_b_chunks]
        logs.append(f"[{ts()}] Method B 切分完成，共 {len(chunks_B)} 个 Chunk；开始构建向量/BM25 索引（此阶段可能较慢）...")
        yield "\n".join(logs), gr.update(), "⏳ Method B 索引构建中...", build_system_status_markdown(), "{}"
        b_index_start = time.time()
        retriever_B.build_index(chunks_B)
        logs.append(f"[{ts()}] Method B 索引写入完成，耗时 {time.time() - b_index_start:.2f}s。")
        elapsed = time.time() - start
        summary = f"chunk数 A/B: {len(chunks_A)}/{len(chunks_B)} | 文档数: {len(doc_names)} | 耗时: {elapsed:.2f}s"
        SYSTEM_STATE["index_ready"] = True
        SYSTEM_STATE["index_summary"] = summary
        logs.append(f"[{ts()}] ✅ Method B 构建完成，共生成 {len(chunks_B)} 个 Chunk。")
        logs.append(f"[{ts()}] 📌 索引摘要：{summary}")
        logs.append(f"\n[{ts()}] 🎉 知识库全部构建完毕！请前往【智能问答】测试。")
        log_event(f"建库完成：{summary}")
        yield "\n".join(logs), gr.update(), "✅ 索引就绪", build_system_status_markdown(), "{}"
    except Exception as e:
        SYSTEM_STATE["index_ready"] = False
        SYSTEM_STATE["index_summary"] = f"构建失败: {e}"
        logs.append(f"\n[{ts()}] ❌ 索引构建失败：{e}")
        logs.append(f"[{ts()}] 👉 建议：检查 PDF 是否损坏、chunk_size 是否过大，或先使用单个 PDF 测试。")
        log_event(f"建库失败：{e}")
        yield (
            "\n".join(logs),
            gr.update(),
            "❌ 索引构建失败",
            build_system_status_markdown(),
            success_error_payload("INDEX_BUILD_FAILED", str(e), "检查 PDF 与参数后重试")
        )
def load_demo_knowledge_base():
    log_event("收到 Demo 数据加载请求。")
    ready, init_error = ensure_retrieval_engines()
    if not ready:
        log_event(f"Demo 加载失败：检索引擎未就绪。原因: {init_error}")
        return (
            f"❌ 检索引擎初始化失败：{init_error}",
            gr.update(),
            "❌ Demo 加载失败",
            build_system_status_markdown(),
            success_error_payload("ENGINE_INIT_FAILED", init_error, "检查环境后重试")
        )
    if not os.path.exists(DEMO_CHUNKS_PATH):
        message = f"未找到 Demo 数据文件：{DEMO_CHUNKS_PATH}"
        log_event(message)
        return (
            f"❌ {message}",
            gr.update(),
            "❌ Demo 加载失败",
            build_system_status_markdown(),
            success_error_payload("DEMO_FILE_MISSING", message, "确认仓库内 demo_chunks.json 是否存在")
        )
    try:
        start = time.time()
        with open(DEMO_CHUNKS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        chunks_A = payload.get("chunks_A") or []
        chunks_B = payload.get("chunks_B") or []
        if not chunks_A and not chunks_B:
            raise ValueError("Demo 文件中缺少 chunks_A/chunks_B 数据")
        if not chunks_A:
            chunks_A = chunks_B
        if not chunks_B:
            chunks_B = chunks_A
        retriever_A.build_index(chunks_A)
        retriever_B.build_index(chunks_B)
        doc_names = sorted(
            {
                line.replace("[Source: ", "").replace("]", "")
                for chunk in chunks_A + chunks_B
                for line in chunk.split("\n", 1)[:1]
                if line.startswith("[Source: ")
            }
        )
        dropdown_choices = ["🌍 全局检索 (混合所有文档)"] + doc_names
        elapsed = time.time() - start
        summary = f"Demo 模式 | chunk数 A/B: {len(chunks_A)}/{len(chunks_B)} | 文档数: {len(doc_names)} | 耗时: {elapsed:.2f}s"
        SYSTEM_STATE["index_ready"] = True
        SYSTEM_STATE["index_summary"] = summary
        logs = [
            f"[{ts()}] ⚡ 已加载内置 Demo 数据（预热弹药库）。",
            f"[{ts()}] 该模式用于快速体验系统布局与问答流程，正式测试请再执行完整 PDF 建库。",
            f"[{ts()}] 📌 索引摘要：{summary}",
        ]
        log_event(f"Demo 加载完成：{summary}")
        return "\n".join(logs), gr.update(choices=dropdown_choices, value=dropdown_choices[0]), "✅ Demo 索引就绪", build_system_status_markdown(), "{}"
    except Exception as e:
        SYSTEM_STATE["index_ready"] = False
        SYSTEM_STATE["index_summary"] = f"Demo 加载失败: {e}"
        log_event(f"Demo 加载失败：{e}")
        return (
            f"❌ Demo 加载失败：{e}",
            gr.update(),
            "❌ Demo 加载失败",
            build_system_status_markdown(),
            success_error_payload("DEMO_LOAD_FAILED", str(e), "请检查 demo_chunks.json 格式")
        )
def find_latest_dashboard_csv() -> Optional[str]:
    candidates = []
    for folder in ["record", "记录"]:
        candidates.extend(glob.glob(os.path.join(folder, "*.csv")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)
def load_dashboard_data():
    required_cols = [
        "A_Hit", "A_MRR", "A_Faith", "A_Rel", "A_ROUGE_L",
        "B_Hit", "B_MRR", "B_Faith", "B_Rel", "B_ROUGE_L",
    ]
    latest_csv = find_latest_dashboard_csv()
    if latest_csv is None:
        SYSTEM_STATE["dashboard_summary"] = "未找到 record/*.csv 或 记录/*.csv"
        log_event("仪表盘加载：未找到可用 CSV。")
        placeholder = pd.DataFrame({"提示": ["暂无评测数据，请先运行 main.py 压测"]})
        return placeholder, "⚠️ 未读取到 CSV 文件", build_system_status_markdown(), "{}"
    try:
        df = pd.read_csv(latest_csv)
        missing_cols = [c for c in required_cols if c not in df.columns]
        msg = f"✅ 已读取: {latest_csv} | 记录数: {len(df)} | 缺失列: {', '.join(missing_cols) if missing_cols else '无'}"
        SYSTEM_STATE["dashboard_summary"] = msg.replace("✅ ", "")
        log_event(f"仪表盘加载成功：{msg}")
        mask_A = ~((df["A_Faith"] == 10) & (df["A_Rel"] == 0))
        mask_B = ~((df["B_Faith"] == 10) & (df["B_Rel"] == 0))
        metrics_df = pd.DataFrame(
            {
                "评测维度 (核心指标)": [
                    "1. 有效召回率 (Hit@5)",
                    "2. 平均倒数排名 (MRR@5)",
                    "3. 真实忠实度 (排除拒答白卷)",
                    "4. 智能拒答率 (找不到则不乱编)",
                    "5. 综合相关性 (0-10)",
                    "6. 专家对齐度 (ROUGE-L)",
                ],
                "Method A (固定切分)": [
                    f"{df['A_Hit'].mean() * 100:.1f}%",
                    f"{df['A_MRR'].mean():.3f}",
                    f"{df.loc[mask_A, 'A_Faith'].mean():.2f}",
                    f"{(1 - mask_A.sum() / len(df)) * 100:.1f}%",
                    f"{df['A_Rel'].mean():.2f}",
                    f"{df['A_ROUGE_L'].mean():.4f}",
                ],
                "Method B (BBox 降噪)": [
                    f"{df['B_Hit'].mean() * 100:.1f}%",
                    f"{df['B_MRR'].mean():.3f}",
                    f"{df.loc[mask_B, 'B_Faith'].mean():.2f}",
                    f"{(1 - mask_B.sum() / len(df)) * 100:.1f}%",
                    f"{df['B_Rel'].mean():.2f}",
                    f"{df['B_ROUGE_L'].mean():.4f}",
                ],
            }
        )
        return metrics_df, msg, build_system_status_markdown(), "{}"
    except Exception as e:
        SYSTEM_STATE["dashboard_summary"] = f"读取失败: {e}"
        log_event(f"仪表盘加载失败：{e}")
        err_payload = success_error_payload("DASHBOARD_READ_FAILED", str(e), "检查 CSV 格式后重试")
        return pd.DataFrame({"错误": [f"读取数据失败: {e}"]}), f"❌ 读取数据失败：{e}", build_system_status_markdown(), err_payload
# ==========================================
# Gradio 页面
# ==========================================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)
with gr.Blocks(title="企业级 RAG 评测系统") as demo:
    gr.Markdown(
        """
<div style="text-align:center; margin: 8px 0 14px 0;">
  <h1>🚀 企业级 RAG 智能问答控制台</h1>
  <p>单页模式：问答 + 建库 + 评测大盘，一屏演示全流程</p>
</div>
"""
    )
    system_status_bar = gr.Markdown(value=build_system_status_markdown())

    gr.Markdown("### 📚 知识库注入")
    gr.Markdown("上传 PDF 后，系统将依次构建 Method A / Method B 双路索引。")
    with gr.Row():
        with gr.Column(scale=2):
            file_upload = gr.File(label="上传 PDF 文献集合", file_count="multiple", file_types=[".pdf"])
            chunk_size_num = gr.Number(value=400, label="Target Chunk Size (字符)")
            build_index_btn = gr.Button("🔨 解析并构建向量/BM25混合索引", variant="primary")
            load_demo_btn = gr.Button("⚡ 先看看效果（加载 Demo 数据）", variant="secondary")
            qa_hint = gr.Markdown("建库完成后可直接在下方【智能交互】提问。")
        with gr.Column(scale=3):
            index_log = gr.Textbox(lines=14, label="实时构建日志", interactive=False)
            with gr.Accordion("🧯 建库错误详情（JSON）", open=False):
                build_error_json = gr.Code(language="json", label="Build Error Payload", value="{}")

    gr.Markdown("### 💬 智能交互")
    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### ⚙️ 检索控制台")
            doc_selector = gr.Dropdown(
                choices=["🌍 全局检索 (混合所有文档)"],
                value="🌍 全局检索 (混合所有文档)",
                label="🎯 检索范围（支持单文档隔离）",
                interactive=True,
            )
            strategy_radio = gr.Radio(
                choices=["Method A (传统固定切分)", "Method B (物理 BBox 降噪)"],
                value="Method B (物理 BBox 降噪)",
                label="检索策略",
            )
            top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top-K 召回数量")
            clear_btn = gr.Button("🧹 清空会话与溯源", variant="secondary")
            qa_index_status = gr.Markdown("⚠️ 索引未就绪")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=420, label="RAG 专家助手")
            msg_input = gr.Textbox(
                placeholder="请输入您关于文献的问题，按 Enter 发送...",
                label="知识库提问",
            )
            with gr.Accordion("🔍 检索证据（RRF Top Chunks）", open=False):
                source_display = gr.Textbox(lines=10, interactive=False, label="Evidence Chains")
            with gr.Accordion("🧯 问答错误详情（JSON）", open=False):
                chat_error_json = gr.Code(language="json", label="Chat Error Payload", value="{}")

    gr.Markdown("### 📊 自动化评测监控大盘")
    gr.Markdown("系统将自动读取 `record/` 或 `记录/` 目录下最新 CSV。")
    with gr.Row():
        refresh_btn = gr.Button("🔄 刷新最新大盘数据", variant="primary")
        dashboard_status = gr.Textbox(label="大盘刷新状态", interactive=False)
    results_df = gr.Dataframe(label="企业级 RAG 核心指标多维对比看板", interactive=False)
    with gr.Accordion("🧯 大盘错误详情（JSON）", open=False):
        dashboard_error_json = gr.Code(language="json", label="Dashboard Error Payload", value="{}")

    build_index_btn.click(
        build_knowledge_base,
        inputs=[file_upload, chunk_size_num],
        outputs=[index_log, doc_selector, qa_index_status, system_status_bar, build_error_json],
    )
    load_demo_btn.click(
        load_demo_knowledge_base,
        inputs=None,
        outputs=[index_log, doc_selector, qa_index_status, system_status_bar, build_error_json],
    )
    msg_input.submit(
        chat_with_rag,
        inputs=[msg_input, chatbot, strategy_radio, top_k_slider, doc_selector],
        outputs=[msg_input, chatbot, source_display, chat_error_json, system_status_bar],
    )
    clear_btn.click(
        lambda: ("", [], "", "{}"),
        inputs=None,
        outputs=[msg_input, chatbot, source_display, chat_error_json],
    )
    refresh_btn.click(
        load_dashboard_data,
        inputs=None,
        outputs=[results_df, dashboard_status, system_status_bar, dashboard_error_json],
        queue=False,
    )
    demo.load(
        load_dashboard_data,
        inputs=None,
        outputs=[results_df, dashboard_status, system_status_bar, dashboard_error_json],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme,
    )
