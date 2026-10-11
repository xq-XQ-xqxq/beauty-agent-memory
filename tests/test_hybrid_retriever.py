"""混合检索器单元测试。

测试策略：
- RRF 融合逻辑：用 FakeRetriever 模拟 BM25 和向量结果，验证融合排序正确
- 接口兼容性：HybridRetriever.search 返回 RetrievedDoc 列表
- 容错：向量检索失败时回退到纯 BM25
- 真实集成测试（需 API Key）：验证端到端检索
"""

import os
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_retriever import HybridRetriever, RRF_K, FusionResult
from knowledge_retriever import KnowledgeRetriever, RetrievedDoc
from vector_retriever import VectorRetriever


# ---------------------------------------------------------------------------
# Fake 组件：避免测试时调用 embedding API
# ---------------------------------------------------------------------------

class FakeBM25Retriever:
    """模拟 KnowledgeRetriever，返回预设的 BM25 结果。"""

    def __init__(self, results: list[RetrievedDoc]):
        self._results = results
        self.doc_count = len(results)

    def search(self, query, doc_type=None, top_k=5):
        # 模拟真实行为：空查询返回空列表
        if not query or not query.strip():
            return []
        return self._results[:top_k]


class FakeVectorRetriever:
    """模拟 VectorRetriever，返回预设的向量结果。"""

    def __init__(self, results: list[RetrievedDoc], is_ready: bool = True):
        self._results = results
        self.is_ready = is_ready
        self.doc_count = len(results)

    def search(self, query, doc_type=None, top_k=5):
        # 模拟真实行为：索引未就绪或空查询返回空列表
        if not self.is_ready or not query or not query.strip():
            return []
        return self._results[:top_k]


def _make_doc(doc_id: str, title: str = "") -> RetrievedDoc:
    return RetrievedDoc(
        doc_id=doc_id,
        doc_type="product",
        title=title or doc_id,
        content="content_" + doc_id,
        metadata={},
        score=0.0,
    )


# ---------------------------------------------------------------------------
# RRF 融合逻辑测试
# ---------------------------------------------------------------------------

def test_rrf_fusion_doc_in_both_ranks_higher():
    """同时出现在 BM25 和向量中的文档，RRF 分数应高于只出现在单路的。"""
    # BM25 排名: A(1), B(2), C(3)
    bm25 = [_make_doc("A"), _make_doc("B"), _make_doc("C")]
    # 向量排名: B(1), A(2), D(3)
    vector = [_make_doc("B"), _make_doc("A"), _make_doc("D")]

    hybrid = HybridRetriever(
        bm25_retriever=FakeBM25Retriever(bm25),
        vector_retriever=FakeVectorRetriever(vector),
    )

    results = hybrid.search_with_detail("test", top_k=10)
    scores = {r.doc.doc_id: r.rrf_score for r in results}

    # A: BM25#1 + 向量#2 = 1/(60+1) + 1/(60+2)
    score_a = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 2)
    # B: BM25#2 + 向量#1 = 1/(60+2) + 1/(60+1)  → 与 A 相同
    score_b = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
    # C: 仅 BM25#3 = 1/(60+3)
    score_c = 1.0 / (RRF_K + 3)
    # D: 仅 向量#3 = 1/(60+3)
    score_d = 1.0 / (RRF_K + 3)

    assert abs(scores["A"] - score_a) < 1e-9
    assert abs(scores["B"] - score_b) < 1e-9
    assert abs(scores["C"] - score_c) < 1e-9
    assert abs(scores["D"] - score_d) < 1e-9

    # 双路命中的 A、B 分数应高于单路命中的 C、D
    assert scores["A"] > scores["C"]
    assert scores["B"] > scores["D"]


def test_rrf_fusion_rank_ordering():
    """RRF 分数应随排名降低而递减。"""
    bm25 = [_make_doc("A"), _make_doc("B"), _make_doc("C")]
    # 向量只有 A，让 B、C 仅靠 BM25
    vector = [_make_doc("A")]

    hybrid = HybridRetriever(
        bm25_retriever=FakeBM25Retriever(bm25),
        vector_retriever=FakeVectorRetriever(vector),
    )

    results = hybrid.search_with_detail("test", top_k=10)
    scores = {r.doc.doc_id: r.rrf_score for r in results}

    # A 在双路，B 在 BM25#2，C 在 BM25#3
    assert scores["A"] > scores["B"] > scores["C"]


def test_rrf_k_constant_is_60():
    """RRF k 值应为 60（Elasticsearch 默认）。"""
    assert RRF_K == 60


# ---------------------------------------------------------------------------
# 接口兼容性测试
# ---------------------------------------------------------------------------

def test_search_returns_retrieved_doc_list():
    """HybridRetriever.search 应返回 RetrievedDoc 列表。"""
    bm25 = [_make_doc("A")]
    vector = [_make_doc("A")]
    hybrid = HybridRetriever(
        bm25_retriever=FakeBM25Retriever(bm25),
        vector_retriever=FakeVectorRetriever(vector),
    )

    results = hybrid.search("test", top_k=5)
    assert isinstance(results, list)
    assert all(isinstance(r, RetrievedDoc) for r in results)


def test_search_respects_top_k():
    """返回结果不超过 top_k。"""
    bm25 = [_make_doc(f"doc_{i}") for i in range(20)]
    vector = [_make_doc(f"doc_{i}") for i in range(20)]
    hybrid = HybridRetriever(
        bm25_retriever=FakeBM25Retriever(bm25),
        vector_retriever=FakeVectorRetriever(vector),
    )

    results = hybrid.search("test", top_k=3)
    assert len(results) <= 3


def test_search_by_type_returns_four_types():
    """search_by_type 应返回 4 种 doc_type 的字典。"""
    bm25 = [_make_doc("A")]
    vector = [_make_doc("A")]
    hybrid = HybridRetriever(
        bm25_retriever=FakeBM25Retriever(bm25),
        vector_retriever=FakeVectorRetriever(vector),
    )

    result = hybrid.search_by_type("test", top_k=3)
    assert set(result.keys()) == {"product", "ingredient", "faq", "policy"}


def test_explain_returns_detail():
    """explain 应返回带融合详情的字典列表。"""
    bm25 = [_make_doc("A")]
    vector = [_make_doc("A")]
    hybrid = HybridRetriever(
        bm25_retriever=FakeBM25Retriever(bm25),
        vector_retriever=FakeVectorRetriever(vector),
    )

    details = hybrid.explain("test", top_k=5)
    assert len(details) == 1
    assert details[0]["doc_id"] == "A"
    assert "rrf_score" in details[0]
    assert "bm25_rank" in details[0]
    assert "vector_rank" in details[0]


# ---------------------------------------------------------------------------
# 容错测试
# ---------------------------------------------------------------------------

def test_vector_not_ready_falls_back_to_bm25():
    """向量索引未就绪时，应回退到纯 BM25 结果。"""
    bm25 = [_make_doc("A"), _make_doc("B")]
    hybrid = HybridRetriever(
        bm25_retriever=FakeBM25Retriever(bm25),
        vector_retriever=FakeVectorRetriever([], is_ready=False),
    )

    results = hybrid.search("test", top_k=5)
    result_ids = {r.doc_id for r in results}
    # 只有 BM25 的结果
    assert result_ids == {"A", "B"}


def test_empty_query_returns_empty():
    """空查询应返回空列表。"""
    bm25 = [_make_doc("A")]
    vector = [_make_doc("A")]
    hybrid = HybridRetriever(
        bm25_retriever=FakeBM25Retriever(bm25),
        vector_retriever=FakeVectorRetriever(vector),
    )

    results = hybrid.search("", top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# 真实集成测试（需要 DASHSCOPE_API_KEY）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="需要 DASHSCOPE_API_KEY 才能运行真实检索测试",
)
def test_real_hybrid_search_returns_results():
    """端到端测试：真实知识库混合检索应返回非空结果。"""
    hybrid = HybridRetriever()
    assert hybrid.vector.is_ready

    results = hybrid.search("敏感肌泛红用什么", doc_type="product", top_k=3)
    assert len(results) > 0
    assert all(isinstance(r, RetrievedDoc) for r in results)

    # 混合检索结果应包含 BM25 和向量的并集
    bm25_ids = {r.doc_id for r in hybrid.bm25.search("敏感肌泛红用什么", doc_type="product", top_k=10)}
    hybrid_ids = {r.doc_id for r in results}
    # 至少有一个 BM25 结果出现在混合结果中
    assert bm25_ids & hybrid_ids


@pytest.mark.skipif(
    not os.environ.get("DASHSCOPE_API_KEY"),
    reason="需要 DASHSCOPE_API_KEY 才能运行真实检索测试",
)
def test_real_vector_search_semantic_match():
    """向量检索应能通过语义匹配找到相关文档（同义词场景）。"""
    vector = VectorRetriever()
    # "脸发红" 和 "泛红" 是同义词，向量检索应能命中修护类产品
    results = vector.search("脸发红发烫", doc_type="product", top_k=5)
    titles = " ".join(r.title for r in results)
    # 至少包含一个修护/舒缓类产品
    assert any(kw in titles for kw in ["修护", "舒缓", "积雪草", "VB5"])
