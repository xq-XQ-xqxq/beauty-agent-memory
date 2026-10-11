"""RAG 检索质量对比评测：BM25 vs 向量 vs 混合（RRF）。

评测指标：
- Recall@K：相关文档中被检索到的比例（K=3, 5）
- Precision@K：检索结果中相关文档的比例（K=3, 5）
- MRR（Mean Reciprocal Rank）：第一个相关文档的排名倒数的平均值

Golden Set 设计原则：
- 覆盖 4 种 doc_type（product/ingredient/faq/policy）
- 包含 BM25 优势场景（精确关键词如产品名、成分名）
- 包含向量优势场景（同义词、口语化表达、语义改写）
- 包含混合优势场景（需要同时利用关键词和语义）

运行：
    python batch_test_rag_comparison.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybrid_retriever import HybridRetriever
from knowledge_retriever import KnowledgeRetriever
from vector_retriever import VectorRetriever


# ---------------------------------------------------------------------------
# Golden Set：查询 → 期望命中的 doc_id 列表
# ---------------------------------------------------------------------------
# 每条记录包含：query（查询）、doc_type（检索类型）、relevant_ids（相关文档 ID）

GOLDEN_SET: list[dict[str, Any]] = [
    # ---- BM25 优势场景：精确关键词 ----
    {
        "query": "水杨酸清洁面膜",
        "doc_type": "product",
        "relevant_ids": ["prod_003"],
        "note": "产品全名精确匹配，BM25 应命中",
    },
    {
        "query": "烟酰胺",
        "doc_type": "ingredient",
        "relevant_ids": ["ing_001"],
        "note": "成分名精确匹配，BM25 应命中",
    },
    {
        "query": "包邮政策",
        "doc_type": "policy",
        "relevant_ids": ["pol_001"],
        "note": "政策标题精确匹配",
    },
    # ---- 向量优势场景：同义词/语义改写 ----
    {
        "query": "脸发红发烫怎么护理",
        "doc_type": "product",
        "relevant_ids": ["prod_001", "prod_008"],  # 积雪草面膜、VB5修护乳
        "note": "泛红的同义词表达，向量应命中修护类产品",
    },
    {
        "query": "怀孕了还能涂护肤品吗",
        "doc_type": "faq",
        "relevant_ids": ["faq_005"],
        "note": "孕妇的口语化表达，向量应命中孕妇 FAQ",
    },
    {
        "query": "BHA 成分",
        "doc_type": "ingredient",
        "relevant_ids": ["ing_004"],  # 水杨酸（别名 BHA）
        "note": "通过别名 BHA 找水杨酸，向量应命中",
    },
    {
        "query": "脸上起皮干燥",
        "doc_type": "product",
        "relevant_ids": ["prod_004", "prod_006"],  # 神经酰胺面霜、玻尿酸水乳
        "note": "干燥的语义表达，向量应命中保湿类产品",
    },
    # ---- 混合优势场景：关键词 + 语义 ----
    {
        "query": "敏感肌祛痘",
        "doc_type": "product",
        "relevant_ids": ["prod_003", "prod_008"],  # 水杨酸(祛痘) + VB5(敏感肌修护)
        "note": "需同时理解敏感肌(语义)和祛痘(关键词)",
    },
    {
        "query": "抗皱紧致精华",
        "doc_type": "product",
        "relevant_ids": ["prod_007"],  # A醇抗皱精华
        "note": "功效关键词 + 产品类型，混合检索应稳定命中",
    },
    {
        "query": "搓泥了怎么办",
        "doc_type": "faq",
        "relevant_ids": ["faq_004"],
        "note": "口语化问题，混合检索应命中",
    },
    {
        "query": "多久能看到美白效果",
        "doc_type": "faq",
        "relevant_ids": ["faq_003"],
        "note": "见效周期的语义改写",
    },
    {
        "query": "退货运费",
        "doc_type": "policy",
        "relevant_ids": ["pol_001", "pol_003"],  # 包邮政策 + 退换货时效
        "note": "跨政策检索，混合应覆盖更多相关文档",
    },
]


# ---------------------------------------------------------------------------
# 评测指标
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """单个查询的检索结果。"""
    query: str
    doc_type: str
    relevant_ids: list[str]
    retrieved_ids: list[str]
    method: str

    @property
    def recall_at_3(self) -> float:
        return self._recall(3)

    @property
    def recall_at_5(self) -> float:
        return self._recall(5)

    @property
    def precision_at_3(self) -> float:
        return self._precision(3)

    @property
    def precision_at_5(self) -> float:
        return self._precision(5)

    @property
    def mrr(self) -> float:
        """第一个相关文档的排名倒数。"""
        for rank, doc_id in enumerate(self.retrieved_ids, start=1):
            if doc_id in self.relevant_ids:
                return 1.0 / rank
        return 0.0

    def _recall(self, k: int) -> float:
        if not self.relevant_ids:
            return 0.0
        retrieved_at_k = set(self.retrieved_ids[:k])
        hits = retrieved_at_k & set(self.relevant_ids)
        return len(hits) / len(self.relevant_ids)

    def _precision(self, k: int) -> float:
        if k == 0:
            return 0.0
        retrieved_at_k = self.retrieved_ids[:k]
        hits = [d for d in retrieved_at_k if d in self.relevant_ids]
        return len(hits) / k


@dataclass
class MethodSummary:
    """单个检索方法的汇总指标。"""
    method: str
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    query_results: list[QueryResult] = field(default_factory=list)

    def compute(self) -> None:
        n = len(self.query_results)
        if n == 0:
            return
        self.recall_at_3 = sum(q.recall_at_3 for q in self.query_results) / n
        self.recall_at_5 = sum(q.recall_at_5 for q in self.query_results) / n
        self.precision_at_3 = sum(q.precision_at_3 for q in self.query_results) / n
        self.precision_at_5 = sum(q.precision_at_5 for q in self.query_results) / n
        self.mrr = sum(q.mrr for q in self.query_results) / n


# ---------------------------------------------------------------------------
# 评测主流程
# ---------------------------------------------------------------------------

def run_evaluation() -> dict[str, MethodSummary]:
    """运行三种检索方法的评测。"""
    bm25 = KnowledgeRetriever()
    vector = VectorRetriever()
    hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector)

    methods = {
        "BM25": bm25,
        "Vector": vector,
        "Hybrid(RRF)": hybrid,
    }

    summaries: dict[str, MethodSummary] = {}

    for method_name, retriever in methods.items():
        summary = MethodSummary(method=method_name)
        for case in GOLDEN_SET:
            results = retriever.search(
                case["query"],
                doc_type=case["doc_type"],
                top_k=5,
            )
            retrieved_ids = [r.doc_id for r in results]
            summary.query_results.append(QueryResult(
                query=case["query"],
                doc_type=case["doc_type"],
                relevant_ids=case["relevant_ids"],
                retrieved_ids=retrieved_ids,
                method=method_name,
            ))
        summary.compute()
        summaries[method_name] = summary

    return summaries


def print_comparison(summaries: dict[str, MethodSummary]) -> None:
    """打印对比表格。"""
    print("\n" + "=" * 80)
    print("RAG 检索质量对比评测：BM25 vs 向量 vs 混合(RRF)")
    print("=" * 80)
    print(f"测试查询数：{len(GOLDEN_SET)}")
    print()

    # 汇总表
    header = f"{'指标':<20} {'BM25':<12} {'Vector':<12} {'Hybrid(RRF)':<12}"
    print(header)
    print("-" * 56)
    for metric in ["recall_at_3", "recall_at_5", "precision_at_3", "precision_at_5", "mrr"]:
        row = f"{metric:<20}"
        for method in ["BM25", "Vector", "Hybrid(RRF)"]:
            value = getattr(summaries[method], metric)
            row += f" {value:<12.4f}"
        print(row)

    # 逐查询详情
    print("\n" + "-" * 80)
    print("逐查询详情：")
    print("-" * 80)
    for case in GOLDEN_SET:
        print(f"\n查询: {case['query']}  [{case['doc_type']}]")
        print(f"  期望: {case['relevant_ids']}")
        print(f"  说明: {case['note']}")
        for method in ["BM25", "Vector", "Hybrid(RRF)"]:
            qr = next(q for q in summaries[method].query_results if q.query == case["query"])
            print(f"  {method:<14} → {qr.retrieved_ids}")


def save_report(summaries: dict[str, MethodSummary], output_path: Path) -> None:
    """保存评测报告为 JSON。"""
    report = {
        "query_count": len(GOLDEN_SET),
        "golden_set": GOLDEN_SET,
        "methods": {},
    }
    for method_name, summary in summaries.items():
        report["methods"][method_name] = {
            "recall_at_3": summary.recall_at_3,
            "recall_at_5": summary.recall_at_5,
            "precision_at_3": summary.precision_at_3,
            "precision_at_5": summary.precision_at_5,
            "mrr": summary.mrr,
            "queries": [
                {
                    "query": q.query,
                    "doc_type": q.doc_type,
                    "relevant_ids": q.relevant_ids,
                    "retrieved_ids": q.retrieved_ids,
                    "recall@3": q.recall_at_3,
                    "recall@5": q.recall_at_5,
                    "precision@3": q.precision_at_3,
                    "precision@5": q.precision_at_5,
                    "mrr": q.mrr,
                }
                for q in summary.query_results
            ],
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n评测报告已保存：{output_path}")


def main() -> None:
    summaries = run_evaluation()
    print_comparison(summaries)
    save_report(summaries, Path(__file__).parent / "data" / "rag_comparison_report.json")


if __name__ == "__main__":
    main()
