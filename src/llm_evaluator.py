import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RAGEvaluator:
    def __init__(
        self,
        generator_id="Qwen/Qwen3.5-2B",
        judge_id="Qwen/Qwen3.5-9B",
        local_files_only=False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[LLM] 加载生成模型: {generator_id}")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generator_id, local_files_only=local_files_only)
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            generator_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=local_files_only,
        )

        print(f"[LLM] 加载评估模型: {judge_id}")
        self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_id, local_files_only=local_files_only)
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            judge_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=local_files_only,
        )
        print("[LLM] 模型加载完成")

    def generate_answer(self, query, context):
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
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        text = self.gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.gen_tokenizer([text], return_tensors="pt").to(self.device)

        outputs = self.gen_model.generate(
            model_inputs.input_ids,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.gen_tokenizer.eos_token_id,
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
        return self.gen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def evaluate_as_judge(self, query, context, generated_answer, ground_truth):
        ans_lower = generated_answer.lower()
        if "cannot answer" in ans_lower or "不知道" in ans_lower or ans_lower.strip() == "":
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
            {"role": "user", "content": (
                f"Context: {context}\nQuestion: {query}\n"
                f"Ground Truth: {ground_truth}\nAI Answer: {generated_answer}"
            )},
        ]

        text = self.judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text += "F_SCORE:"

        model_inputs = self.judge_tokenizer([text], return_tensors="pt").to(self.device)
        outputs = self.judge_model.generate(
            model_inputs.input_ids,
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            pad_token_id=self.judge_tokenizer.eos_token_id,
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)]
        judgment = self.judge_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        full_judgment = "F_SCORE:" + judgment

        f_match = re.search(r"F_SCORE:\s*(\d+)", full_judgment, re.IGNORECASE)
        r_match = re.search(r"R_SCORE:\s*(\d+)", full_judgment, re.IGNORECASE)

        if f_match and r_match:
            f_score = int(f_match.group(1))
            r_score = int(r_match.group(1))
        else:
            scores = re.findall(r"\d+", full_judgment)
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
        def lcs(x, y):
            m, n = len(x), len(y)
            table = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0 or j == 0:
                        table[i][j] = 0
                    elif x[i - 1] == y[j - 1]:
                        table[i][j] = table[i - 1][j - 1] + 1
                    else:
                        table[i][j] = max(table[i - 1][j], table[i][j - 1])
            return table[m][n]

        gen_tokens = re.findall(r"\w+", generated.lower())
        gt_tokens = re.findall(r"\w+", ground_truth.lower())
        if not gen_tokens or not gt_tokens:
            return 0.0

        lcs_len = lcs(gen_tokens, gt_tokens)
        precision = lcs_len / len(gen_tokens)
        recall = lcs_len / len(gt_tokens)
        if precision + recall == 0:
            return 0.0

        f1_score = (2 * precision * recall) / (precision + recall)
        return round(f1_score, 4)
