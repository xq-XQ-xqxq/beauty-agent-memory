"""混合检索器：BM25 关键词检索 + 向量语义检索 + RRF 融合。

融合算法 — Reciprocal Rank Fusion (RRF)：
    score(d) = Σ 1 / (k + rank_i(d))
    其中 k 是常数（默认 60，Elasticsearch 默认值），rank_i 是文档 d 在第 i 路检索中的排名。

为什么用 RRF 而不是加权平均：
- BM25 分数和向量余弦分数的尺度完全不同，直接加权需要调权重
- RRF 只看排名，不看原始分数，天然兼容不同评分尺度
- 无需调参，效果稳定，是工业界混合检索的标配

面试考点：
- RRF 的 k 值含义：k 越大，排名靠后的文档权重衰减越慢；k=60 是经验最优
- 为什么需要混合检索：BM25 精确匹配关键词但不懂语义；向量懂语义但可能漏精确匹配
- 什么场景下向量更好：同义词、口语化表达（如"脸泛红"→ 检索到"敏感肌修护"）
- 什么场景下 BM25 更好：产品名、成分名等专有名词精确匹配
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_retriever import KnowledgeRetriever, RetrievedDoc
from vector_retriever import VectorRetriever


# RRF 常数（Elasticsearch 默认值 60）
RRF_K = 60


@dataclass
class FusionResult:
    """RRF 融合后的单条结果。"""
    doc: RetrievedDoc
    rrf_score: float
    bm25_rank: int | None
    vector_rank: int | None


class HybridRetriever:
    """混合检索器：BM25 + 向量 + RRF。

    接口与 KnowledgeRetriever 完全兼容：
        retriever = HybridRetriever()
        results = retriever.search(query, doc_type="product", top_k=3)
    """

    def __init__(
        self,
        documents_path: Path | str | None = None,
        bm25_retriever: KnowledgeRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        self._bm25 = bm25_retriever or KnowledgeRetriever(documents_path)
        self._vector = vector_retriever or VectorRetriever(documents_path)
        self._rrf_k = rrf_k

    @property
    def bm25(self) -> KnowledgeRetriever:
        return self._bm25

    @property
    def vector(self) -> VectorRetriever:
        return self._vector

    @property
    def doc_count(self) -> int:
        return self._bm25.doc_count

    # ------------------------------------------------------------------
    # 核心检索方法
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedDoc]:
        """混合检索，返回 RetrievedDoc 列表（与 KnowledgeRetriever 接口一致）。"""
        fusion_results = self.search_with_detail(query, doc_type=doc_type, top_k=top_k)
        return [fr.doc for fr in fusion_results]

    def search_with_detail(
        self,
        query: str,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[FusionResult]:
        """混合检索，返回带融合详情的结果（含 RRF 分数、各路排名）。"""
        # 两路检索各取更多候选项，融合后再截断到 top_k
        candidate_k = max(top_k * 3, 10)

        bm25_results = self._bm25.search(query, doc_type=doc_type, top_k=candidate_k)
        vector_results = self._vector.search(query, doc_type=doc_type, top_k=candidate_k)

        # 构建 doc_id → rank 映射
        bm25_ranks: dict[str, int] = {
            d.doc_id: rank for rank, d in enumerate(bm25_results, start=1)
        }
        vector_ranks: dict[str, int] = {
            d.doc_id: rank for rank, d in enumerate(vector_results, start=1)
        }

        # 合并所有候选 doc_id
        all_doc_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())

        # 构建 doc_id → RetrievedDoc 映射（优先用 BM25 的，因为它 metadata 完整）
        doc_map: dict[str, RetrievedDoc] = {}
        for d in bm25_results:
            doc_map[d.doc_id] = d
        for d in vector_results:
            if d.doc_id not in doc_map:
                doc_map[d.doc_id] = d

        # RRF 融合
        fused: list[FusionResult] = []
        for doc_id in all_doc_ids:
            bm25_rank = bm25_ranks.get(doc_id)
            vector_rank = vector_ranks.get(doc_id)

            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += 1.0 / (self._rrf_k + bm25_rank)
            if vector_rank is not None:
                rrf_score += 1.0 / (self._rrf_k + vector_rank)

            fused.append(FusionResult(
                doc=doc_map[doc_id],
                rrf_score=rrf_score,
                bm25_rank=bm25_rank,
                vector_rank=vector_rank,
            ))

        # 按 RRF 分数降序，取 top_k
        fused.sort(key=lambda x: x.rrf_score, reverse=True)
        return fused[:top_k]

    def search_by_type(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, list[RetrievedDoc]]:
        """按 doc_type 分组返回检索结果（与 KnowledgeRetriever 接口一致）。"""
        return {
            dt: self.search(query, doc_type=dt, top_k=top_k)
            for dt in ("product", "ingredient", "faq", "policy")
        }

    # ------------------------------------------------------------------
    # 诊断方法
    # ------------------------------------------------------------------

    def explain(self, query: str, doc_type: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        """返回融合过程的详细解释（用于调试和评测）。"""
        results = self.search_with_detail(query, doc_type=doc_type, top_k=top_k)
        return [
            {
                "doc_id": fr.doc.doc_id,
                "title": fr.doc.title,
                "rrf_score": round(fr.rrf_score, 6),
                "bm25_rank": fr.bm25_rank,
                "vector_rank": fr.vector_rank,
                "doc_type": fr.doc.doc_type,
            }
            for fr in results
        ]
