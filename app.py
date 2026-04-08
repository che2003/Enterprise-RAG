import gradio as gr
import pandas as pd
import os
import time
import glob
import json

# 导入底层引擎
from src.data_pipeline import DataPipeline
from src.hybrid_retriever import HybridRetriever
from src.llm_evaluator import RAGEvaluator

print("==================================================")
print("🚀 正在启动企业级 RAG WebUI 服务，请稍候...")
print("==================================================")

# ==========================================
# 1. 全局单例初始化
# ==========================================
pipeline = DataPipeline()
retriever_A = HybridRetriever()
retriever_B = HybridRetriever()

# 💡 建议：如果只是跑 UI 演示，可以不传 judge_id 以节省显存
try:
    evaluator = RAGEvaluator(
        generator_id="Qwen/Qwen3.5-2B",
        judge_id=r"D:\models\Qwen3.5-9B"
    )
    print("✅ [系统] LLM 引擎加载成功！")
except Exception as e:
    print(f"⚠️ [系统] LLM 引擎加载失败，请检查显存或路径: {e}")
    evaluator = None


SYSTEM_STATE = {
    "embedding_ready": True,   # retriever 单例可用即视作 embedding 引擎就绪
    "model_ready": evaluator is not None,
    "index_ready": False,
    "index_summary": "尚未构建索引",
    "dashboard_summary": "尚未读取评测 CSV"
}


def build_system_status_markdown():
    """统一状态栏：展示 embedding/model/index 初始化与运行状态"""
    embedding_flag = "✅" if SYSTEM_STATE["embedding_ready"] else "❌"
    model_flag = "✅" if SYSTEM_STATE["model_ready"] else "❌"
    index_flag = "✅" if SYSTEM_STATE["index_ready"] else "⚠️"

    return (
        "### 🟢 系统统一状态栏\n"
        f"- Embedding 引擎：{embedding_flag}\n"
        f"- LLM 生成引擎：{model_flag}\n"
        f"- 检索索引：{index_flag}\n"
        f"- 索引摘要：{SYSTEM_STATE['index_summary']}\n"
        f"- 大盘读取状态：{SYSTEM_STATE['dashboard_summary']}"
    )


# ==========================================
# 2. 核心交互函数绑定
# ==========================================
def chat_with_rag(user_message, history, strategy, top_k, target_doc):
    """处理用户提问并返回答案与溯源上下文"""
    if not user_message.strip():
        return "", history, "请输入有效问题。"

    time.sleep(0.5)  # 模拟思考延迟

    # 将前端的防串库选项传递给底层引擎
    doc_filter = target_doc if target_doc != "🌍 全局检索 (混合所有文档)" else None

    # 💡 核心修复：加入 try-except 块，优雅捕获“未建库”的错误
    try:
        if "Method A" in strategy:
            context_str, chunks_list = retriever_A.hybrid_search_rrf(user_message, top_k=int(top_k),
                                                                     target_doc_name=doc_filter)
        else:
            context_str, chunks_list = retriever_B.hybrid_search_rrf(user_message, top_k=int(top_k),
                                                                     target_doc_name=doc_filter)
    except ValueError as e:
        error_payload = {
            "error_code": "INDEX_NOT_READY",
            "message": str(e),
            "action": "请先上传 PDF 并点击构建索引"
        }
        error_msg = f"⚠️ 检索失败 [{error_payload['error_code']}]：{error_payload['action']}"
        history.append((user_message, error_msg))
        return "", history, json.dumps(error_payload, ensure_ascii=False, indent=2)
    except Exception as e:
        error_payload = {
            "error_code": "RAG_RUNTIME_ERROR",
            "message": str(e),
            "action": "请重试；如仍失败，请重新构建索引并检查模型加载状态"
        }
        error_msg = f"❌ 检索过程中发生系统异常 [{error_payload['error_code']}]"
        history.append((user_message, error_msg))
        return "", history, json.dumps(error_payload, ensure_ascii=False, indent=2)

    if not chunks_list:
        ans = "知识库中暂未检索到相关内容，请尝试更换问题或扩充知识库。"
        context_str = "未命中任何文档片段。"
    else:
        if evaluator:
            ans = evaluator.generate_answer(user_message, context_str)
        else:
            ans = "模型未加载，这是 Mock 答案：根据检索结果，无法确定准确答案。"

    history.append((user_message, ans))
    return "", history, context_str


def build_knowledge_base(file_objs, chunk_size):
    """处理用户上传的 PDF 并构建双路索引，同时更新下拉菜单"""
    if not file_objs:
        yield "⚠️ 未检测到文件，请先上传 PDF。", gr.update(), build_system_status_markdown(), "⚠️ 索引未就绪"
        return

    log_messages = []
    start_time = time.time()
    file_paths = [f.name for f in file_objs]
    doc_names = [os.path.basename(p) for p in file_paths]

    # 动态生成防串库下拉菜单的选项
    dropdown_choices = ["🌍 全局检索 (混合所有文档)"] + doc_names

    log_messages.append(f"📦 收到 {len(file_paths)} 篇文献: {', '.join(doc_names)}")
    log_messages.append(f"⚙️ 当前切分块大小设置: {chunk_size}")
    yield "\n".join(log_messages), gr.update(choices=dropdown_choices, value="🌍 全局检索 (混合所有文档)"), build_system_status_markdown(), "⏳ 索引构建中..."

    log_messages.append("\n>> 正在构建 Method A (基础固定切分)...")
    yield "\n".join(log_messages), gr.update(), build_system_status_markdown(), "⏳ 索引构建中..."

    chunks_A = pipeline.naive_fixed_chunking(file_paths, chunk_size=int(chunk_size), overlap=50)
    retriever_A.build_index(chunks_A)
    log_messages.append(f"✅ Method A 构建完成，共生成 {len(chunks_A)} 个 Chunk。")
    yield "\n".join(log_messages), gr.update(), build_system_status_markdown(), "⏳ 索引构建中..."

    log_messages.append("\n>> 正在构建 Method B (物理 BBox 降噪切分)...")
    yield "\n".join(log_messages), gr.update(), build_system_status_markdown(), "⏳ 索引构建中..."

    chunks_B = pipeline.bbox_layout_chunking(file_paths, target_chunk_size=int(chunk_size))
    retriever_B.build_index(chunks_B)
    elapsed = time.time() - start_time
    summary = f"chunk数 A/B: {len(chunks_A)}/{len(chunks_B)} | 文档数: {len(doc_names)} | 耗时: {elapsed:.2f}s"
    SYSTEM_STATE["index_ready"] = True
    SYSTEM_STATE["index_summary"] = summary
    log_messages.append(f"✅ Method B 构建完成，共生成 {len(chunks_B)} 个 Chunk。")
    log_messages.append(f"📌 索引摘要：{summary}")
    log_messages.append("\n🎉 知识库全部构建完毕！请前往【智能问答】测试。")
    yield "\n".join(log_messages), gr.update(), build_system_status_markdown(), "✅ 索引就绪"


def load_dashboard_data():
    """读取 record 目录下的最新 CSV 并转化为 DataFrame 展示"""
    required_cols = [
        "A_Hit", "A_MRR", "A_Faith", "A_Rel", "A_ROUGE_L",
        "B_Hit", "B_MRR", "B_Faith", "B_Rel", "B_ROUGE_L"
    ]
    csv_files = glob.glob("record/*.csv")
    if not csv_files:
        SYSTEM_STATE["dashboard_summary"] = "未找到 record/*.csv"
        return (
            pd.DataFrame({"提示": ["暂无评测数据，请先运行 main.py 压测"]}),
            "⚠️ 未读取到 CSV 文件",
            build_system_status_markdown()
        )

    latest_csv = max(csv_files, key=os.path.getctime)

    try:
        df = pd.read_csv(latest_csv)
        missing_cols = [col for col in required_cols if col not in df.columns]
        dashboard_msg = (
            f"✅ 已读取: {latest_csv} | 记录数: {len(df)} | "
            f"缺失列: {', '.join(missing_cols) if missing_cols else '无'}"
        )
        SYSTEM_STATE["dashboard_summary"] = dashboard_msg.replace("✅ ", "")

        # 修正指标剔除规则，依赖裁判给出的 F=10, R=0 (诚实拒答标志)
        mask_A = ~((df['A_Faith'] == 10) & (df['A_Rel'] == 0))
        mask_B = ~((df['B_Faith'] == 10) & (df['B_Rel'] == 0))

        data = {
            "评测维度 (核心指标)": [
                "1. 有效召回率 (Hit@5)",
                "2. 平均倒数排名 (MRR@5)",
                "3. 真实忠实度 (排除拒答白卷)",
                "4. 智能拒答率 (找不到则不乱编)",
                "5. 综合相关性 (0-10)",
                "6. 专家对齐度 (ROUGE-L)"
            ],
            "Method A (固定切分)": [
                f"{df['A_Hit'].mean() * 100:.1f}%",
                f"{df['A_MRR'].mean():.3f}",
                f"{df.loc[mask_A, 'A_Faith'].mean():.2f}",
                f"{(1 - mask_A.sum() / len(df)) * 100:.1f}%",
                f"{df['A_Rel'].mean():.2f}",
                f"{df['A_ROUGE_L'].mean():.4f}"
            ],
            "Method B (BBox 降噪)": [
                f"{df['B_Hit'].mean() * 100:.1f}%",
                f"{df['B_MRR'].mean():.3f}",
                f"{df.loc[mask_B, 'B_Faith'].mean():.2f}",
                f"{(1 - mask_B.sum() / len(df)) * 100:.1f}%",
                f"{df['B_Rel'].mean():.2f}",
                f"{df['B_ROUGE_L'].mean():.4f}"
            ]
        }
        return pd.DataFrame(data), dashboard_msg, build_system_status_markdown()
    except Exception as e:
        SYSTEM_STATE["dashboard_summary"] = f"读取失败: {e}"
        return (
            pd.DataFrame({"错误": [f"读取数据失败: {e}"]}),
            f"❌ 读取数据失败：{e}",
            build_system_status_markdown()
        )


# ==========================================
# 3. 构建 WebUI 前端页面
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

# 💡 移除了这里的 theme 参数，避免 UserWarning
with gr.Blocks(title="企业级 RAG 评测系统") as demo:
    gr.Markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1>🚀 异构双模型企业级 RAG 智能问答系统</h1>
        <p>基于物理 BBox 排版降噪、混合检索 (RRF) 与大模型异步裁判 (LLM-as-a-Judge) 驱动</p>
    </div>
    """)
    system_status_bar = gr.Markdown(value=build_system_status_markdown())

    with gr.Tabs():
        # ------------------------------------------
        # Tab 1: 智能问答区
        # ------------------------------------------
        with gr.Tab("💬 智能交互中心"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 检索控制台")

                    doc_selector = gr.Dropdown(
                        choices=["🌍 全局检索 (混合所有文档)"],
                        value="🌍 全局检索 (混合所有文档)",
                        label="🎯 检索范围 (单篇防串库隔离)",
                        interactive=True
                    )

                    strategy_radio = gr.Radio(
                        choices=["Method A (传统固定切分)", "Method B (物理 BBox 降噪)"],
                        value="Method B (物理 BBox 降噪)",
                        label="切换引擎策略 (对比体验)"
                    )
                    top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K 召回数量")
                    clear_btn = gr.Button("🧹 清空对话与溯源", variant="secondary")

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=450, label="RAG 专家助手")
                    msg_input = gr.Textbox(placeholder="请输入您关于文献的问题，按 Enter 键发送...", label="知识库提问")
                    qa_index_status = gr.Markdown("⚠️ 索引未就绪")

                    with gr.Accordion("🔍 查看底层检索溯源 (Evidence Chains)", open=False):
                        source_display = gr.Textbox(lines=8, label="Top 召回切片 (Chunks)", interactive=False)

            # 绑定问答事件
            msg_input.submit(
                chat_with_rag,
                inputs=[msg_input, chatbot, strategy_radio, top_k_slider, doc_selector],
                outputs=[msg_input, chatbot, source_display]
            )
            # 绑定清空事件
            clear_btn.click(lambda: ("", [], ""), inputs=None, outputs=[msg_input, chatbot, source_display])

        # ------------------------------------------
        # Tab 2: 知识库管理
        # ------------------------------------------
        with gr.Tab("📚 私有知识库注入"):
            gr.Markdown("将 PDF 论文拖拽至此，系统将自动使用双路切分策略构建底层知识图谱。")
            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(label="上传 PDF 文献集合", file_count="multiple", file_types=[".pdf"])
                    chunk_size_num = gr.Number(value=400, label="Target Chunk Size (字符)")
                    build_index_btn = gr.Button("🔨 解析并构建向量/BM25混合索引", variant="primary")
                with gr.Column():
                    index_log = gr.Textbox(lines=12, label="实时构建日志 (观察降噪过程)", interactive=False)

            # 绑定建库事件
            build_index_btn.click(
                build_knowledge_base,
                inputs=[file_upload, chunk_size_num],
                outputs=[index_log, doc_selector, system_status_bar, qa_index_status]
            )

        # ------------------------------------------
        # Tab 3: 大盘数据看板
        # ------------------------------------------
        with gr.Tab("📊 自动化评测监控大盘"):
            gr.Markdown("### 🏆 异构裁判 (Qwen3.5-9B) 给出的最终系统评估报告")
            gr.Markdown("系统会自动读取 `record/` 目录下的最新 `.csv` 文件，清洗并渲染评测指标。")

            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新最新大盘数据", variant="primary")
                dashboard_status = gr.Textbox(label="大盘刷新状态", interactive=False)

            results_df = gr.Dataframe(label="企业级 RAG 核心指标多维对比看板", interactive=False)

            refresh_btn.click(load_dashboard_data, inputs=None, outputs=[results_df, dashboard_status, system_status_bar])
            demo.load(load_dashboard_data, inputs=None, outputs=[results_df, dashboard_status, system_status_bar])

# ==========================================
# 4. 启动服务
# ==========================================
if __name__ == "__main__":
    # 💡 核心修复：添加了 .queue() 防止卡死，theme 参数移至此处
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=theme
    )
