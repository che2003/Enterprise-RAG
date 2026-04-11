import os

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    def __init__(self, embed_model_name="BAAI/bge-small-en-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_model = self._load_embedding_model(embed_model_name)
        self.texts = []
        self.faiss_index = None
        self.bm25_index = None

    def _load_embedding_model(self, embed_model_name):
        local_model_path = os.getenv("RAG_EMBED_MODEL_PATH", "").strip()
        allow_online_fallback = os.getenv("RAG_ALLOW_ONLINE_MODEL", "0").strip() == "1"

        candidates = [c for c in [local_model_path, embed_model_name] if c]
        load_errors = []

        for candidate in candidates:
            try:
                return SentenceTransformer(candidate, device=self.device, local_files_only=True)
            except TypeError:
                try:
                    return SentenceTransformer(candidate, device=self.device)
                except Exception as e:
                    load_errors.append(f"{candidate}: {e}")
            except Exception as e:
                load_errors.append(f"{candidate}: {e}")

        if allow_online_fallback:
            for candidate in candidates:
                try:
                    return SentenceTransformer(candidate, device=self.device)
                except Exception as e:
                    load_errors.append(f"{candidate} (online): {e}")

        raise RuntimeError(
            "Embedding 模型离线加载失败。请先将模型下载到本地并设置 "
            "RAG_EMBED_MODEL_PATH，或设置 RAG_ALLOW_ONLINE_MODEL=1 允许在线拉取。"
            f" 详细错误: {' | '.join(load_errors)}"
        )

    def build_index(self, chunks):
        self.texts = chunks
        if not self.texts:
            return

        embeddings = self.embed_model.encode(self.texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        self.faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.faiss_index.add(embeddings)

        tokenized_corpus = [doc.lower().split(" ") for doc in self.texts]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def hybrid_search_rrf(self, query, top_k=5, rrf_k=60, target_doc_name=None, mode="hybrid"):
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("索引未构建")

        valid_indices = set(range(len(self.texts)))
        if target_doc_name:
            source_tag = f"[Source: {target_doc_name}]"
            valid_indices = {i for i, text in enumerate(self.texts) if text.startswith(source_tag)}
            if not valid_indices:
                valid_indices = set(range(len(self.texts)))

        candidate_size = min(60, len(valid_indices))
        if candidate_size == 0:
            return "", []

        query_vec = np.array(self.embed_model.encode([query])).astype("float32")
        search_k = min(200, len(self.texts))
        _, faiss_indices = self.faiss_index.search(query_vec, search_k)
        faiss_rank_list = [idx for idx in faiss_indices[0].tolist() if idx in valid_indices][:candidate_size]

        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_rank_list = np.argsort(bm25_scores)[::-1].tolist()
        bm25_rank_list = [idx for idx in bm25_rank_list if idx in valid_indices][:candidate_size]

        if mode == "dense":
            sorted_indices = faiss_rank_list[:top_k]
        elif mode == "sparse":
            sorted_indices = bm25_rank_list[:top_k]
        else:
            rrf_scores = {}
            for rank, doc_idx in enumerate(faiss_rank_list):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            for rank, doc_idx in enumerate(bm25_rank_list):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
            sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

        fused_contexts = [self.texts[i] for i in sorted_indices]
        return "\n\n=== Retrieved Top Chunk ===\n\n".join(fused_contexts), fused_contexts
