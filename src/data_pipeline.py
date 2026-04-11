import os
import re

import fitz
import requests


class DataPipeline:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.raw_papers_dir = os.path.join(data_dir, "raw_papers")
        os.makedirs(self.raw_papers_dir, exist_ok=True)

    def fetch_qasper_sample(self, num_papers=10):
        print(f"正在通过 Hugging Face API 拉取 {num_papers} 篇 QASper 论文...")
        api_url = (
            "https://datasets-server.huggingface.co/rows?"
            f"dataset=allenai/qasper&config=qasper&split=validation&offset=0&length={num_papers}"
        )

        pdf_paths = []
        eval_qas = []

        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            api_data = response.json()

            for item in api_data["rows"]:
                sample_paper = item["row"]
                arxiv_id = sample_paper["id"]
                doc_name = f"{arxiv_id}.pdf"

                pdf_path = os.path.join(self.raw_papers_dir, doc_name)
                if not os.path.exists(pdf_path):
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    pdf_response = requests.get(pdf_url, timeout=30)
                    if pdf_response.status_code == 200:
                        with open(pdf_path, "wb") as f:
                            f.write(pdf_response.content)
                    else:
                        continue
                pdf_paths.append(pdf_path)

                questions = sample_paper["qas"].get("question", [])
                answers_list = sample_paper["qas"].get("answers", [])

                for q, ans_data in zip(questions, answers_list):
                    gt_text = "N/A"
                    try:
                        ans_dict = ans_data.get("answer", {})
                        if isinstance(ans_dict, dict):
                            free_forms = ans_dict.get("free_form_answer", [])
                            spans = ans_dict.get("extractive_spans", [])
                            if free_forms and free_forms[0]:
                                gt_text = free_forms[0]
                            elif spans and spans[0]:
                                gt_text = " ".join(spans[0])
                        elif isinstance(ans_dict, list) and ans_dict:
                            first_ans = ans_dict[0]
                            if first_ans.get("free_form_answer"):
                                gt_text = first_ans["free_form_answer"]
                            elif first_ans.get("extractive_spans"):
                                gt_text = " ".join(first_ans["extractive_spans"])
                    except Exception:
                        continue

                    if gt_text != "N/A" and gt_text.strip():
                        cleaned_gt = re.sub(r"BIBREF\d*", "", gt_text)
                        cleaned_gt = re.sub(r"\$.*?\$", "", cleaned_gt)
                        cleaned_gt = re.sub(r"\s+", " ", cleaned_gt).strip()

                        if cleaned_gt:
                            eval_qas.append((q, cleaned_gt, doc_name))

            return pdf_paths, eval_qas
        except Exception as e:
            print(f"数据拉取失败: {e}")
            return [], []

    def naive_fixed_chunking(self, pdf_paths, chunk_size=400, overlap=50):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        chunks = []
        for path in pdf_paths:
            doc_name = os.path.basename(path)
            try:
                doc = fitz.open(path)
                text = "".join([page.get_text() for page in doc])
                start = 0
                while start < len(text):
                    chunks.append(f"[Source: {doc_name}]\n{text[start:start + chunk_size]}")
                    start += chunk_size - overlap
            except Exception as e:
                print(f"读取 {path} 失败，已跳过: {e}")
        return chunks

    def bbox_layout_chunking(self, pdf_paths, target_chunk_size=400):
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]
        chunks = []
        for path in pdf_paths:
            doc_name = os.path.basename(path)
            try:
                doc = fitz.open(path)
                current_chunk = ""
                for page in doc:
                    for _, _, _, _, _, _, _, cleaned_text in self._get_layout_sorted_blocks(page):
                        current_chunk += cleaned_text + " \n"
                        if len(current_chunk) >= target_chunk_size:
                            chunks.append(f"[Source: {doc_name}]\n{current_chunk.strip()}")
                            current_chunk = ""
                if current_chunk:
                    chunks.append(f"[Source: {doc_name}]\n{current_chunk.strip()}")
            except Exception as e:
                print(f"解析 {path} 失败，已跳过: {e}")
        return chunks

    def _get_layout_sorted_blocks(self, page, full_width_threshold=0.7):
        page_width = page.rect.width
        center_x = page_width / 2.0

        blocks = [b for b in page.get_text("blocks") if b[6] == 0]

        full_width_blocks = []
        left_column = []
        right_column = []

        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b

            cleaned_text = text.replace("-\n", "").replace("\n", " ").strip()
            if len(cleaned_text) < 20:
                continue

            alpha_count = sum(c.isalpha() for c in cleaned_text)
            if len(cleaned_text) > 0 and (alpha_count / len(cleaned_text)) < 0.15:
                continue

            block_width = x1 - x0
            block_center_x = (x0 + x1) / 2.0
            processed_block = (x0, y0, x1, y1, text, block_no, block_type, cleaned_text)

            if block_width > page_width * full_width_threshold:
                full_width_blocks.append(processed_block)
            elif block_center_x < center_x:
                left_column.append(processed_block)
            else:
                right_column.append(processed_block)

        for group in (full_width_blocks, left_column, right_column):
            group.sort(key=lambda b: b[1])

        return full_width_blocks + left_column + right_column
