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


def error_payload(error_code: str, message: str, action: str) -> str:
    return json.dumps(
        {"error_code": error_code, "message": message, "action": action},
        ensure_ascii=False,
        indent=2,
    )


def gate_panel_updates(phase: str):
    if phase == "eval_ready":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "✅ 评测就绪：索引已构建，可以使用监控/大盘/Case分析能力。",
            gr.update(interactive=True),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        "🚧 数据准备期：请先上传 PDF 并构建索引，系统才会开放核心业务页面。",
        gr.update(interactive=False),
    )


def ensure_retrieval_engines() -> Tuple[bool, str]:
    global pipeline, retriever_A, retriever_B
    if pipeline is not None and retriever_A is not None and retriever_B is not None:
        return True, ""

    SYSTEM_STATE["engine_status"] = "检索引擎冷启动中..."
    try:
        pipeline = DataPipeline()
        retriever_A = HybridRetriever()
        retriever_B = HybridRetriever()
        SYSTEM_STATE["embedding_ready"] = True
        SYSTEM_STATE["engine_status"] = "检索引擎已就绪"
        return True, ""
    except Exception as e:
        SYSTEM_STATE["embedding_ready"] = False
        SYSTEM_STATE["engine_status"] = f"检索引擎初始化失败: {e}"
        return False, str(e)


def ensure_evaluator_engine() -> Tuple[bool, str]:
    global evaluator, model_init_attempted
    if evaluator is not None:
        return True, ""
    if model_init_attempted:
        return False, "模型初始化已失败，请检查模型路径或显存"

    model_init_attempted = True
    SYSTEM_STATE["engine_status"] = "LLM 引擎冷启动中..."
    try:
        evaluator = RAGEvaluator(
            generator_id="Qwen/Qwen3.5-2B",
            judge_id=r"D:\models\Qwen3.5-9B",
        )
        SYSTEM_STATE["model_ready"] = True
        SYSTEM_STATE["engine_status"] = "LLM 引擎已就绪"
        return True, ""
    except Exception as e:
        SYSTEM_STATE["model_ready"] = False
        SYSTEM_STATE["engine_status"] = f"LLM 初始化失败: {e}"
        return False, str(e)


def chat_with_rag(
    user_message: str,
    history: List[Tuple[str, str]],
    strategy: str,
    top_k: int,
    target_doc: str,
):
    if not user_message.strip():
        return "", history, "请输入有效问题。", "{}", build_system_status_markdown()

    if not SYSTEM_STATE["index_ready"]:
        payload = error_payload("INDEX_NOT_READY", "索引尚未就绪", "请先在初始化面板构建索引")
        history.append((user_message, "⚠️ 当前处于数据准备期，请先建库。"))
        return "", history, "暂无溯源数据。", payload, build_system_status_markdown()

    time.sleep(0.2)
    doc_filter = target_doc if target_doc != "🌍 全局检索 (混合所有文档)" else None

    ready, init_error = ensure_retrieval_engines()
    if not ready:
        payload = error_payload("ENGINE_INIT_FAILED", init_error, "请检查依赖与日志后重启服务")
        history.append((user_message, "❌ 检索引擎初始化失败"))
        return "", history, "引擎未就绪，无法提供溯源。", payload, build_system_status_markdown()

    try:
        retriever = retriever_A if "Method A" in strategy else retriever_B
        context_str, chunks_list = retriever.hybrid_search_rrf(
            user_message, top_k=int(top_k), target_doc_name=doc_filter
        )
    except ValueError as e:
        payload = error_payload("INDEX_NOT_READY", str(e), "请先上传 PDF 并点击构建索引")
        history.append((user_message, "⚠️ 当前未建索引，请先建库。"))
        return "", history, "暂无溯源数据。", payload, build_system_status_markdown()
    except Exception as e:
        payload = error_payload("RAG_RUNTIME_ERROR", str(e), "请重试；失败则重建索引")
        history.append((user_message, "❌ 检索过程中发生系统异常。"))
        return "", history, "系统异常，溯源不可用。", payload, build_system_status_markdown()

    if not chunks_list:
        answer = "知识库中暂未检索到相关内容，请尝试更换问题或扩充知识库。"
        context_str = "未命中任何文档片段。"
        payload = "{}"
    else:
        model_ready, _ = ensure_evaluator_engine()
        if model_ready and evaluator is not None:
            answer = evaluator.generate_answer(user_message, context_str)
            payload = "{}"
        else:
            answer = "模型冷启动失败，当前返回检索摘要模式。请检查模型路径或显存。"
            payload = error_payload("MODEL_NOT_READY", "生成模型未就绪", "检查模型路径与显存后重试")

    history.append((user_message, answer))
    return "", history, context_str, payload, build_system_status_markdown()


def build_knowledge_base_with_gate(file_objs, chunk_size):
    if not file_objs:
        init_vis, core_vis, gate_msg, gate_btn = gate_panel_updates("data_prep")
        yield (
            "⚠️ 未检测到文件，请先上传 PDF。",
            gr.update(),
            "⚠️ 索引未就绪",
            build_system_status_markdown(),
            "{}",
            "data_prep",
            init_vis,
            core_vis,
            gate_msg,
            gate_btn,
        )
        return

    try:
        chunk_size = int(chunk_size)
    except Exception:
        init_vis, core_vis, gate_msg, gate_btn = gate_panel_updates("data_prep")
        yield (
            "⚠️ chunk_size 非法，请输入有效整数（例如 400）。",
            gr.update(),
            "⚠️ 索引未就绪",
            build_system_status_markdown(),
            error_payload("INVALID_CHUNK_SIZE", "chunk_size 非法", "请输入整数并重试"),
            "data_prep",
            init_vis,
            core_vis,
            gate_msg,
            gate_btn,
        )
        return

    ready, init_error = ensure_retrieval_engines()
    if not ready:
        init_vis, core_vis, gate_msg, gate_btn = gate_panel_updates("data_prep")
        yield (
            f"❌ 检索引擎初始化失败：{init_error}",
            gr.update(),
            "❌ 索引构建失败",
            build_system_status_markdown(),
            error_payload("ENGINE_INIT_FAILED", init_error, "检查环境后重试"),
            "data_prep",
            init_vis,
            core_vis,
            gate_msg,
            gate_btn,
        )
        return

    start = time.time()
    logs = []
    file_paths = [f.name for f in file_objs]
    doc_names = [os.path.basename(p) for p in file_paths]
    dropdown_choices = ["🌍 全局检索 (混合所有文档)"] + doc_names

    logs.append(f"📦 收到 {len(file_paths)} 篇文献: {', '.join(doc_names)}")
    logs.append(f"⚙️ 当前切分块大小设置: {chunk_size}")
    init_vis, core_vis, gate_msg, gate_btn = gate_panel_updates("data_prep")
    yield (
        "\n".join(logs),
        gr.update(choices=dropdown_choices, value=dropdown_choices[0]),
        "⏳ 索引构建中...",
        build_system_status_markdown(),
        "{}",
        "data_prep",
        init_vis,
        core_vis,
        gate_msg,
        gate_btn,
    )

    try:
        logs.append("\n>> 正在构建 Method A (基础固定切分)...")
        yield "\n".join(logs), gr.update(), "⏳ 索引构建中...", build_system_status_markdown(), "{}", "data_prep", init_vis, core_vis, gate_msg, gate_btn

        chunks_A = pipeline.naive_fixed_chunking(file_paths, chunk_size=chunk_size, overlap=50)
        retriever_A.build_index(chunks_A)
        logs.append(f"✅ Method A 构建完成，共生成 {len(chunks_A)} 个 Chunk。")
        yield "\n".join(logs), gr.update(), "⏳ 索引构建中...", build_system_status_markdown(), "{}", "data_prep", init_vis, core_vis, gate_msg, gate_btn

        logs.append("\n>> 正在构建 Method B (物理 BBox 降噪切分)...")
        yield "\n".join(logs), gr.update(), "⏳ 索引构建中...", build_system_status_markdown(), "{}", "data_prep", init_vis, core_vis, gate_msg, gate_btn

        chunks_B = pipeline.bbox_layout_chunking(file_paths, target_chunk_size=chunk_size)
        retriever_B.build_index(chunks_B)

        elapsed = time.time() - start
        summary = f"chunk数 A/B: {len(chunks_A)}/{len(chunks_B)} | 文档数: {len(doc_names)} | 耗时: {elapsed:.2f}s"
        SYSTEM_STATE["index_ready"] = True
        SYSTEM_STATE["index_summary"] = summary

        logs.append(f"✅ Method B 构建完成，共生成 {len(chunks_B)} 个 Chunk。")
        logs.append(f"📌 索引摘要：{summary}")
        logs.append("\n🎉 初始化完成！点击下方按钮进入核心业务页面。")

        init_vis, core_vis, gate_msg, gate_btn = gate_panel_updates("eval_ready")
        yield (
            "\n".join(logs),
            gr.update(),
            "✅ 索引就绪",
            build_system_status_markdown(),
            "{}",
            "eval_ready",
            init_vis,
            core_vis,
            gate_msg,
            gate_btn,
        )
    except Exception as e:
        SYSTEM_STATE["index_ready"] = False
        SYSTEM_STATE["index_summary"] = f"构建失败: {e}"
        logs.append(f"\n❌ 索引构建失败：{e}")
        logs.append("👉 建议：检查 PDF 是否损坏、chunk_size 是否过大，或先使用单个 PDF 测试。")
        init_vis, core_vis, gate_msg, gate_btn = gate_panel_updates("data_prep")
        yield (
            "\n".join(logs),
            gr.update(),
            "❌ 索引构建失败",
            build_system_status_markdown(),
            error_payload("INDEX_BUILD_FAILED", str(e), "检查 PDF 与参数后重试"),
            "data_prep",
            init_vis,
            core_vis,
            gate_msg,
            gate_btn,
        )


def enter_core_system(phase: str):
    return gate_panel_updates(phase)


def find_latest_dashboard_csv() -> Optional[str]:
    candidates = []
    for folder in ["record", "记录"]:
        candidates.extend(glob.glob(os.path.join(folder, "*.csv")))
    return max(candidates, key=os.path.getctime) if candidates else None


def load_dashboard_data():
    required_cols = [
        "A_Hit", "A_MRR", "A_Faith", "A_Rel", "A_ROUGE_L",
        "B_Hit", "B_MRR", "B_Faith", "B_Rel", "B_ROUGE_L",
    ]

    latest_csv = find_latest_dashboard_csv()
    if latest_csv is None:
        SYSTEM_STATE["dashboard_summary"] = "未找到 record/*.csv 或 记录/*.csv"
        return pd.DataFrame({"提示": ["暂无评测数据，请先运行 main.py 压测"]}), "⚠️ 未读取到 CSV 文件", build_system_status_markdown(), "{}"

    try:
        df = pd.read_csv(latest_csv)
        missing_cols = [c for c in required_cols if c not in df.columns]
        msg = f"✅ 已读取: {latest_csv} | 记录数: {len(df)} | 缺失列: {', '.join(missing_cols) if missing_cols else '无'}"
        SYSTEM_STATE["dashboard_summary"] = msg.replace("✅ ", "")

        mask_A = ~((df["A_Faith"] == 10) & (df["A_Rel"] == 0))
        mask_B = ~((df["B_Faith"] == 10) & (df["B_Rel"] == 0))

        metrics_df = pd.DataFrame({
            "评测维度 (核心指标)": [
                "1. 有效召回率 (Hit@5)", "2. 平均倒数排名 (MRR@5)", "3. 真实忠实度 (排除拒答白卷)",
                "4. 智能拒答率 (找不到则不乱编)", "5. 综合相关性 (0-10)", "6. 专家对齐度 (ROUGE-L)",
            ],
            "Method A (固定切分)": [
                f"{df['A_Hit'].mean() * 100:.1f}%", f"{df['A_MRR'].mean():.3f}", f"{df.loc[mask_A, 'A_Faith'].mean():.2f}",
                f"{(1 - mask_A.sum() / len(df)) * 100:.1f}%", f"{df['A_Rel'].mean():.2f}", f"{df['A_ROUGE_L'].mean():.4f}",
            ],
            "Method B (BBox 降噪)": [
                f"{df['B_Hit'].mean() * 100:.1f}%", f"{df['B_MRR'].mean():.3f}", f"{df.loc[mask_B, 'B_Faith'].mean():.2f}",
                f"{(1 - mask_B.sum() / len(df)) * 100:.1f}%", f"{df['B_Rel'].mean():.2f}", f"{df['B_ROUGE_L'].mean():.4f}",
            ],
        })
        return metrics_df, msg, build_system_status_markdown(), "{}"
    except Exception as e:
        SYSTEM_STATE["dashboard_summary"] = f"读取失败: {e}"
        payload = error_payload("DASHBOARD_READ_FAILED", str(e), "检查 CSV 格式后重试")
        return pd.DataFrame({"错误": [f"读取数据失败: {e}"]}), f"❌ 读取数据失败：{e}", build_system_status_markdown(), payload


def load_case_samples():
    return pd.DataFrame({
        "场景": ["索引未就绪", "单文档隔离检索", "模型降级模式"],
        "预期行为": ["拦截并引导先建库", "仅召回目标文档 chunk", "返回检索摘要并给出错误码"],
    })


theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

with gr.Blocks(title="企业级 RAG 评测系统", theme=theme) as demo:
    app_phase = gr.State("data_prep")

    gr.Markdown(
        """
<div style="text-align:center; margin: 8px 0 14px 0;">
  <h1>🚀 企业级 RAG 评测系统（双阶段防呆模式）</h1>
  <p>阶段1：数据准备期（必须先建库） → 阶段2：评测就绪期（开放监控/大盘/Case分析）</p>
</div>
"""
    )

    system_status_bar = gr.Markdown(value=build_system_status_markdown())

    with gr.Column(visible=True) as init_container:
        gr.Markdown("## 🧭 初始化引导面板\n请先完成 PDF 上传与索引构建，系统才会开放核心业务 Tab。")
        gate_tip = gr.Markdown("🚧 数据准备期：请先上传 PDF 并构建索引，系统才会开放核心业务页面。")

        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(label="上传 PDF 文献集合", file_count="multiple", file_types=[".pdf"])
                chunk_size_num = gr.Number(value=400, label="Target Chunk Size (字符)")
                build_index_btn = gr.Button("🔨 开始初始化（构建双路索引）", variant="primary")
                enter_btn = gr.Button("🚪 进入核心业务页面", interactive=False)
            with gr.Column(scale=3):
                init_log = gr.Textbox(lines=14, label="初始化实时日志", interactive=False)
                init_status = gr.Markdown("⚠️ 索引未就绪")
                with gr.Accordion("🧯 初始化错误详情（JSON）", open=False):
                    init_error_json = gr.Code(language="json", value="{}")

    with gr.Column(visible=False) as core_container:
        with gr.Tabs():
            with gr.Tab("💬 监控中心"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=320):
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
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=480, label="RAG 监控助手")
                        msg_input = gr.Textbox(placeholder="请输入您关于文献的问题，按 Enter 发送...", label="知识库提问")
                        with gr.Accordion("🔍 检索证据（RRF Top Chunks）", open=False):
                            source_display = gr.Textbox(lines=10, interactive=False)
                        with gr.Accordion("🧯 错误详情（JSON）", open=False):
                            chat_error_json = gr.Code(language="json", value="{}")

            with gr.Tab("📊 自动化评测大盘"):
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新最新大盘数据", variant="primary")
                    dashboard_status = gr.Textbox(label="大盘刷新状态", interactive=False)
                results_df = gr.Dataframe(label="企业级 RAG 核心指标对比看板", interactive=False)
                with gr.Accordion("🧯 大盘错误详情（JSON）", open=False):
                    dashboard_error_json = gr.Code(language="json", value="{}")

            with gr.Tab("🧪 Case分析"):
                gr.Markdown("典型场景行为基线（用于验收与回归）。")
                case_df = gr.Dataframe(label="Case 行为检查表", interactive=False)

    build_index_btn.click(
        build_knowledge_base_with_gate,
        inputs=[file_upload, chunk_size_num],
        outputs=[
            init_log,
            doc_selector,
            init_status,
            system_status_bar,
            init_error_json,
            app_phase,
            init_container,
            core_container,
            gate_tip,
            enter_btn,
        ],
    )

    enter_btn.click(
        enter_core_system,
        inputs=[app_phase],
        outputs=[init_container, core_container, gate_tip, enter_btn],
    )

    msg_input.submit(
        chat_with_rag,
        inputs=[msg_input, chatbot, strategy_radio, top_k_slider, doc_selector],
        outputs=[msg_input, chatbot, source_display, chat_error_json, system_status_bar],
    )

    clear_btn.click(lambda: ("", [], "", "{}"), outputs=[msg_input, chatbot, source_display, chat_error_json])

    refresh_btn.click(
        load_dashboard_data,
        outputs=[results_df, dashboard_status, system_status_bar, dashboard_error_json],
        queue=False,
    )
    demo.load(load_dashboard_data, outputs=[results_df, dashboard_status, system_status_bar, dashboard_error_json], queue=False)
    demo.load(load_case_samples, outputs=[case_df], queue=False)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=True)
