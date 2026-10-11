"""知识库 BM25 检索器。

复用 jieba + rank_bm25（与 context_engine 同一技术栈），零额外依赖。
特点：
1. 同义词扩展：美妆领域 20 条硬编码同义词（如"长痘"→"痤疮/痘痘/水杨酸"）
2. doc_type 过滤：可按 product / ingredient / faq / policy 分类型检索
3. <10ms 纯内存检索，零 API 调用
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

jieba.setLogLevel(20)

# ---------------------------------------------------------------------------
# 同义词词典（美妆领域硬编码，0ms 成本）
# ---------------------------------------------------------------------------

SYNONYM_MAP: dict[str, list[str]] = {
    "长痘": ["痤疮", "痘痘", "水杨酸", "祛痘"],
    "痘痘": ["痤疮", "长痘", "水杨酸", "祛痘"],
    "痤疮": ["痘痘", "长痘", "水杨酸"],
    "泛红": ["红", "敏感", "舒缓", "积雪草", "修护"],
    "过敏": ["敏感", "刺痛", "泛红", "不耐受"],
    "干皮": ["干性", "干燥", "保湿", "玻尿酸"],
    "油皮": ["油性", "出油", "控油", "水杨酸"],
    "敏感肌": ["敏感", "泛红", "修护", "积雪草", "神经酰胺"],
    "美白": ["提亮", "烟酰胺", "维C", "亮肤"],
    "提亮": ["美白", "烟酰胺", "维C", "亮肤"],
    "抗老": ["抗皱", "抗初老", "A醇", "视黄醇", "虾青素"],
    "抗皱": ["抗老", "A醇", "视黄醇", "细纹"],
    "细纹": ["皱纹", "抗皱", "A醇"],
    "补水": ["保湿", "玻尿酸", "水润"],
    "保湿": ["补水", "玻尿酸", "水润"],
    "换季": ["敏感", "泛红", "舒缓", "VB5", "积雪草"],
    "孕妇": ["孕期", "怀孕", "禁用", "可用"],
    "退货": ["退款", "售后", "7天"],
    "包邮": ["运费", "物流"],
    "搓泥": ["泥", "使用方法", "吸收"],
}

# 反向索引：同义词 → 标准词
_REVERSE_SYNONYM: dict[str, str] = {}
for standard, aliases in SYNONYM_MAP.items():
    for alias in aliases:
        _REVERSE_SYNONYM[alias] = standard


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDoc:
    doc_id: str
    doc_type: str
    title: str
    content: str
    metadata: dict[str, Any]
    score: float


# ---------------------------------------------------------------------------
# 知识库检索器
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    """BM25 知识库检索器：jieba 分词 + 同义词扩展 + doc_type 过滤。

    用法：
        retriever = KnowledgeRetriever()
        results = retriever.search("我长痘了用什么", doc_type="product", top_k=3)
    """

    def __init__(self, documents_path: Path | str | None = None) -> None:
        if documents_path is None:
            documents_path = Path(__file__).parent / "knowledge_base" / "documents.json"
        with open(documents_path, encoding="utf-8") as f:
            self._documents: list[dict[str, Any]] = json.load(f)

        # 预构建 BM25 索引
        self._corpus = [self._tokenize(self._doc_text(d)) for d in self._documents]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    @staticmethod
    def _doc_text(doc: dict[str, Any]) -> str:
        """把文档展平为可检索文本：标题 + 内容 + 别名。"""
        parts = [doc.get("title", ""), doc.get("content", "")]
        metadata = doc.get("metadata", {})
        aliases = metadata.get("aliases", [])
        if isinstance(aliases, list):
            parts.extend(aliases)
        tags = metadata.get("tags", [])
        if isinstance(tags, list):
            parts.extend(tags)
        return " ".join(parts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.strip() for t in jieba.cut(text) if t.strip() and len(t.strip()) > 1]

    def _expand_query(self, query: str) -> list[str]:
        """同义词扩展：把 query 中匹配到的同义词追加到分词列表。"""
        tokens = self._tokenize(query)
        expanded = list(tokens)
        for token in tokens:
            # 标准词 → 追加同义词
            if token in SYNONYM_MAP:
                expanded.extend(SYNONYM_MAP[token][:3])  # 最多追加 3 个，避免噪声
            # 同义词 → 追加标准词
            if token in _REVERSE_SYNONYM:
                standard = _REVERSE_SYNONYM[token]
                if standard not in expanded:
                    expanded.append(standard)
        return expanded

    def search(
        self,
        query: str,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedDoc]:
        """检索知识库。

        Args:
            query: 用户查询文本
            doc_type: 过滤文档类型（product/ingredient/faq/policy），None 不过滤
            top_k: 返回前 K 条
        """
        if not self._bm25:
            return []

        query_tokens = self._expand_query(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results: list[RetrievedDoc] = []
        for idx, score in ranked:
            doc = self._documents[idx]
            if doc_type and doc.get("doc_type") != doc_type:
                continue
            if score > 0:
                results.append(RetrievedDoc(
                    doc_id=doc.get("doc_id", ""),
                    doc_type=doc.get("doc_type", ""),
                    title=doc.get("title", ""),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=float(score),
                ))
            if len(results) >= top_k:
                break
        return results

    def search_by_type(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, list[RetrievedDoc]]:
        """按 doc_type 分组返回检索结果。"""
        return {
            dt: self.search(query, doc_type=dt, top_k=top_k)
            for dt in ("product", "ingredient", "faq", "policy")
        }

    @property
    def doc_count(self) -> int:
        return len(self._documents)
