import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


class RAGEvaluator:
    def __init__(
        self,
        generator_id="Qwen/Qwen3.5-2B",
        judge_id="Qwen/Qwen3.5-9B",
        local_files_only=False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"🚀 [LLM] 正在加载选手模型 (Generator): {generator_id} ...")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generator_id, local_files_only=local_files_only)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            generator_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=local_files_only,
        )

        print(f"⚖️ [LLM] 正在加载裁判模型 (Judge): {judge_id} ...")
        self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_id, local_files_only=local_files_only)
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            judge_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=local_files_only,
        )
        print("✅ [LLM] 双引擎异构大模型就绪，准备降维打击！")

    def generate_answer(self, query, context):
        """由 2B 小模型负责生成答案 (智能拒答边界 + 部分提取策略)"""
        messages = [
            {"role": "system", "content": (
                "You are an exact extraction bot. You will be provided with retrieved text chunks.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. Answer the question based ONLY on the provided Context. Keep it extremely concise.\n"
                "2. PARTIAL MATCH: If the Context contains only a partial answer, synonyms, or incomplete information, EXTRACT exactly what is there. Do NOT refuse just because it is incomplete.\n"
                "3. DO NOT use introductory phrases. Output ONLY the core entities or facts.\n"
                "4. ZERO HALLUCINATION: ONLY if the Context is COMPLETELY irrelevant or contains NO helpful information, you MUST output exactly: 'I cannot answer'. DO NOT guess.\n\n"
                "=== EXAMPLES ===\n"
                "Context: We experimented with Europarl and MultiUN datasets.\n"
                "Question: Which datasets did they experiment with?\n"
                "Answer: Europarl and MultiUN\n\n"
                "Context: The cat is sleeping on the sofa.\n"
                "Question: What accuracy does the proposed system achieve?\n"
                "Answer: I cannot answer\n"
                "================"
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        text = self.gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.gen_tokenizer([text], return_tensors="pt").to(self.device)

        # 保持 150 Token 限制，确保提取的实体不会被腰斩
        outputs = self.gen_model.generate(
            model_inputs.input_ids, max_new_tokens=150, temperature=0.1, do_sample=True,
            pad_token_id=self.gen_tokenizer.eos_token_id
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
        return self.gen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def evaluate_as_judge(self, query, context, generated_answer, ground_truth):
        """由 9B 大模型结合 Ground Truth 进行开卷裁判 (硬规则拦截 + 绝对确定性生成)"""

        # =====================================================================
        # 💡 硬规则拦截：检测到诚实拒答，直接判定为无幻觉，赋予 F:10, R:0
        # =====================================================================
        ans_lower = generated_answer.lower()
        if "cannot answer" in ans_lower or "不知道" in ans_lower or ans_lower.strip() == "":
            print("\n[Judge Bypass]: 🚨 模型诚实地拒答了！判定为无幻觉，赋分 F:10, R:0")
            return {"Faithfulness": 0, "Relevance": 0}

        messages = [
            {"role": "system", "content": (
                "You are an expert evaluator assessing an AI Answer based on Context and Ground Truth.\n\n"
                "Metric 1: Faithfulness (0-10) - Is it based ONLY on the Context?\n"
                "- 10: Perfectly faithful to context.\n"
                "- 0: Contradicts context or hallucinates.\n\n"
                "Metric 2: Relevance (0-10) - Does it match the Ground Truth?\n"
                "- 10: Perfect factual match to the Ground Truth (ignore conversational filler).\n"
                "- 5: Partial match (AI captures some but not all of the Ground Truth entities or facts).\n"
                "- 0: Completely contradicts or misses the Ground Truth facts entirely.\n\n"
                "Output EXACTLY and ONLY the numbers in this format:\n"
                "F_SCORE: [number]\n"
                "R_SCORE: [number]"
            )},
            {"role": "user",
             "content": f"Context: {context}\nQuestion: {query}\nGround Truth: {ground_truth}\nAI Answer: {generated_answer}"}
        ]

        text = self.judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += "F_SCORE:"  # 物理截流外挂，强迫直接输出数字

        model_inputs = self.judge_tokenizer([text], return_tensors="pt").to(self.device)

        # 关闭采样，确保打分绝对稳定
        outputs = self.judge_model.generate(
            model_inputs.input_ids,
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            pad_token_id=self.judge_tokenizer.eos_token_id
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
        judgment = self.judge_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        full_judgment = "F_SCORE:" + judgment
        print(f"\n[Judge Output]: {full_judgment.replace(chr(10), ' | ')}")

        import re
        f_match = re.search(r'F_SCORE:\s*(\d+)', full_judgment, re.IGNORECASE)
        r_match = re.search(r'R_SCORE:\s*(\d+)', full_judgment, re.IGNORECASE)

        if f_match and r_match:
            f_score = int(f_match.group(1))
            r_score = int(r_match.group(1))
        else:
            scores = re.findall(r'\d+', full_judgment)
            if len(scores) >= 2:
                f_score, r_score = int(scores[0]), int(scores[1])
            elif len(scores) == 1:
                f_score, r_score = int(scores[0]), 0
            else:
                f_score, r_score = 0, 0

        f_score = min(max(f_score, 0), 10)
        r_score = min(max(r_score, 0), 10)

        return {"Faithfulness": f_score, "Relevance": r_score}

    def compute_rouge_l(self, generated, ground_truth):
        def lcs(X, Y):
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])
            return L[m][n]

        gen_tokens = re.findall(r'\w+', generated.lower())
        gt_tokens = re.findall(r'\w+', ground_truth.lower())

        if not gen_tokens or not gt_tokens: return 0.0

        lcs_len = lcs(gen_tokens, gt_tokens)
        precision = lcs_len / len(gen_tokens)
        recall = lcs_len / len(gt_tokens)

        if precision + recall == 0: return 0.0
        f1_score = (2 * precision * recall) / (precision + recall)
        return round(f1_score, 4)