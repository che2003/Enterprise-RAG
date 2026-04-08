import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, embed_model_name='BAAI/bge-small-en-v1.5'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)
        self.texts = []
        self.faiss_index = None
        self.bm25_index = None

    def build_index(self, chunks):
        self.texts = chunks
        if not self.texts: return

        # 1. FAISS
        embeddings = self.embed_model.encode(self.texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        # 2. BM25
        tokenized_corpus = [doc.lower().split(" ") for doc in self.texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def hybrid_search_rrf(self, query, top_k=5, rrf_k=60, target_doc_name=None):
        """
        核心突破：Reciprocal Rank Fusion (RRF) + 物理硬隔离
        将 FAISS 和 BM25 的召回名次进行倒数融合，彻底消除分数尺度的差异。
        新增 target_doc_name 进行硬过滤，杜绝跨文档串库。
        """
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("索引未构建！")

        # 💡 筛选属于该题目的合法 Chunk
        valid_indices = set(range(len(self.texts)))
        if target_doc_name:
            source_tag = f"[Source: {target_doc_name}]"
            valid_indices = {i for i, text in enumerate(self.texts) if text.startswith(source_tag)}
            if not valid_indices:  # 防御性编程：如果没有匹配到，降级为全局搜索
                valid_indices = set(range(len(self.texts)))

        candidate_size = min(60, len(valid_indices))
        if candidate_size == 0: return "", []

        # --- 获取各自的候选名次 ---

        # FAISS：放大搜索池，过滤出合法 Chunk
        query_vec = np.array(self.embed_model.encode([query])).astype("float32")
        search_k = min(200, len(self.texts))
        _, faiss_indices = self.faiss_index.search(query_vec, search_k)
        faiss_rank_list = [idx for idx in faiss_indices[0].tolist() if idx in valid_indices][:candidate_size]

        # BM25：同样只保留合法 Chunk
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_rank_list = np.argsort(bm25_scores)[::-1].tolist()
        bm25_rank_list = [idx for idx in bm25_rank_list if idx in valid_indices][:candidate_size]

        # --- 执行 RRF 分数计算 ---
        rrf_scores = {}

        for rank, doc_idx in enumerate(faiss_rank_list):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, doc_idx in enumerate(bm25_rank_list):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        # --- 按 RRF 得分排序，截取最终的 Top-K ---
        sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        # 组装返回的 Context
        fused_contexts = [self.texts[i] for i in sorted_indices]
        return "\n\n=== RRF Top Chunk ===\n\n".join(fused_contexts), fused_contexts