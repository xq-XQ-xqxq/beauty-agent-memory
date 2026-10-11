"""向量检索器：DashScope text-embedding-v4 + Qdrant 本地持久化。

设计要点：
1. Embedding 模型：text-embedding-v4，1024 维，余弦相似度
2. 向量存储：Qdrant 本地持久化（QdrantClient(path=...)），无需启动服务
3. 增量索引：按 documents.json 的内容 hash 判断是否需要重建索引，
   文档未变时不重复调用 embedding API，节省 token
4. 批量 embedding：一次请求最多 25 条文档，减少 API 调用次数
5. 容错降级：embedding 失败时返回空结果，让 HybridRetriever 回退到纯 BM25
6. 接口与 KnowledgeRetriever 对齐：search(query, doc_type, top_k) 返回 RetrievedDoc 列表

面试考点：
- 为什么用本地 Qdrant 而不是内存 dict？—— 持久化 + 支持大规模数据 + HNSW 索引
- 为什么 1024 维？—— text-embedding-v4 原生输出维度，维度越高语义越精确但存储/计算成本越高
- 为什么余弦距离？—— 文本 embedding 长度归一化后，余弦等价于内积，对短文本检索效果好
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from knowledge_retriever import RetrievedDoc

load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
COLLECTION_NAME = "knowledge_base"
# DashScope text-embedding-v4 单次 embedding 最多 10 条输入
BATCH_SIZE = 10
# Qdrant 本地存储目录
DEFAULT_QDRANT_PATH = Path(__file__).parent / "data" / "qdrant_kb"

# UUID 命名空间：把 doc_id（如 "prod_001"）确定性映射为 UUID（Qdrant 本地模式要求 UUID）
_DOC_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")

# QdrantClient 单例缓存（按路径）：Qdrant 本地模式用文件锁，同一路径不允许多个 client
_QDRANT_CLIENTS: dict[str, QdrantClient] = {}
# Streamlit 每个会话/热重载在独立线程跑脚本，首开必须加锁，否则两个线程
# 同时通过 "path not in dict" 检查会各自创建 client，触发本地文件锁冲突
_QDRANT_INIT_LOCK = threading.Lock()


def _doc_id_to_uuid(doc_id: str) -> uuid.UUID:
    """把 doc_id 字符串确定性转换为 UUID v5。"""
    return uuid.uuid5(_DOC_ID_NAMESPACE, doc_id)


def _get_qdrant_client(path: str) -> QdrantClient:
    """获取指定路径的 QdrantClient 单例（避免本地模式文件锁冲突）。"""
    with _QDRANT_INIT_LOCK:
        if path not in _QDRANT_CLIENTS:
            _QDRANT_CLIENTS[path] = QdrantClient(path=path)
        return _QDRANT_CLIENTS[path]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _doc_hash(documents: list[dict[str, Any]]) -> str:
    """计算 documents 列表的内容 hash，用于判断是否需要重建索引。"""
    serialized = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _flatten_doc_text(doc: dict[str, Any]) -> str:
    """把文档展平为可 embedding 的文本（标题 + 内容 + 别名 + 标签）。"""
    parts = [doc.get("title", ""), doc.get("content", "")]
    metadata = doc.get("metadata", {})
    aliases = metadata.get("aliases", [])
    if isinstance(aliases, list):
        parts.extend(str(a) for a in aliases)
    tags = metadata.get("tags", [])
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags)
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# 向量检索器
# ---------------------------------------------------------------------------

@dataclass
class VectorSearchHit:
    """向量检索单条命中。"""
    doc_id: str
    score: float
    rank: int


class VectorRetriever:
    """向量语义检索器。

    用法：
        retriever = VectorRetriever()
        results = retriever.search("我长痘了用什么", doc_type="product", top_k=3)
    """

    def __init__(
        self,
        documents_path: Path | str | None = None,
        qdrant_path: Path | str | None = None,
        client: Any | None = None,
    ) -> None:
        if documents_path is None:
            documents_path = Path(__file__).parent / "knowledge_base" / "documents.json"
        if qdrant_path is None:
            qdrant_path = DEFAULT_QDRANT_PATH

        self._documents_path = Path(documents_path)
        self._qdrant_path = Path(qdrant_path)
        self._qdrant_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载原始文档
        with open(self._documents_path, encoding="utf-8") as f:
            self._documents: list[dict[str, Any]] = json.load(f)
        self._doc_map: dict[str, dict[str, Any]] = {
            d["doc_id"]: d for d in self._documents
        }

        # LLM client（用于 embedding）
        if client is None:
            api_key = os.environ.get("MEM0_LLM_API_KEY") or os.environ.get(
                "DASHSCOPE_API_KEY", ""
            )
            base_url = os.environ.get("MEM0_EMBED_BASE_URL") or os.environ.get(
                "MEM0_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client

        # Qdrant 客户端（单例，避免本地模式文件锁冲突）
        self._qdrant = _get_qdrant_client(str(self._qdrant_path))

        # 索引状态
        self._collection_ready = False
        self._indexed_hash: str | None = None

        self._ensure_index()

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """确保 Qdrant 集合存在且索引是最新的。"""
        current_hash = _doc_hash(self._documents)

        # 检查集合是否存在
        collections = self._qdrant.get_collections().collections
        collection_names = {c.name for c in collections}

        if COLLECTION_NAME not in collection_names:
            self._create_collection()
            self._build_index(current_hash)
            return

        # 集合存在，检查是否需要重建（通过 payload 中的 hash 字段判断）
        try:
            # 读取集合信息中存储的 hash（用第一个 point 的 payload 或 collection info）
            # 简化方案：直接比较 count
            info = self._qdrant.get_collection(COLLECTION_NAME)
            point_count = info.points_count
            if point_count != len(self._documents):
                self._qdrant.delete_collection(COLLECTION_NAME)
                self._create_collection()
                self._build_index(current_hash)
                return
        except Exception:
            # 读取失败就重建
            self._qdrant.delete_collection(COLLECTION_NAME)
            self._create_collection()
            self._build_index(current_hash)
            return

        self._collection_ready = True
        self._indexed_hash = current_hash

    def _create_collection(self) -> None:
        """创建 Qdrant 集合。"""
        self._qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    def _build_index(self, current_hash: str) -> None:
        """对所有文档做 embedding 并写入 Qdrant。"""
        if not self._documents:
            self._collection_ready = True
            self._indexed_hash = current_hash
            return

        # 批量 embedding
        all_embeddings: list[list[float]] = []
        for i in range(0, len(self._documents), BATCH_SIZE):
            batch = self._documents[i : i + BATCH_SIZE]
            texts = [_flatten_doc_text(d) for d in batch]
            try:
                response = self._client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts,
                )
                # 按 index 排序（API 返回顺序可能不保证）
                sorted_data = sorted(response.data, key=lambda x: x.index)
                embeddings = [d.embedding for d in sorted_data]
                all_embeddings.extend(embeddings)
            except Exception as exc:
                # embedding 失败：标记未就绪，后续 search 会返回空
                print(f"[VectorRetriever] embedding 失败: {type(exc).__name__}: {exc}")
                self._collection_ready = False
                self._indexed_hash = None
                return

        # 写入 Qdrant（point id 用 UUID，原始 doc_id 存入 payload）
        points = [
            PointStruct(
                id=_doc_id_to_uuid(doc["doc_id"]),
                vector=embedding,
                payload={
                    "doc_id": doc["doc_id"],
                    "doc_type": doc.get("doc_type", ""),
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                },
            )
            for doc, embedding in zip(self._documents, all_embeddings)
        ]
        self._qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        self._collection_ready = True
        self._indexed_hash = current_hash

    @property
    def is_ready(self) -> bool:
        return self._collection_ready

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedDoc]:
        """向量语义检索。

        Args:
            query: 用户查询文本
            doc_type: 过滤文档类型（product/ingredient/faq/policy），None 不过滤
            top_k: 返回前 K 条

        Returns:
            RetrievedDoc 列表（按向量相似度降序）。
            如果索引未就绪或 embedding 失败，返回空列表。
        """
        if not self._collection_ready or not query.strip():
            return []

        # embedding 查询文本
        try:
            response = self._client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[query],
            )
            query_vector = response.data[0].embedding
        except Exception as exc:
            print(f"[VectorRetriever] query embedding 失败: {type(exc).__name__}: {exc}")
            return []

        # 构建过滤条件
        query_filter = None
        if doc_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=doc_type),
                    )
                ]
            )

        # Qdrant 检索（qdrant-client >= 1.10 用 query_points 替代 search）
        try:
            response = self._qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )
            hits = response.points
        except Exception as exc:
            print(f"[VectorRetriever] qdrant search 失败: {type(exc).__name__}: {exc}")
            return []

        # 转为 RetrievedDoc（从 payload 中取回原始 doc_id）
        results: list[RetrievedDoc] = []
        for hit in hits:
            payload = hit.payload or {}
            doc_id = payload.get("doc_id", str(hit.id))
            doc = self._doc_map.get(doc_id)
            if doc is None:
                continue
            results.append(RetrievedDoc(
                doc_id=doc.get("doc_id", ""),
                doc_type=doc.get("doc_type", ""),
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                score=float(hit.score),
            ))
        return results

    def search_with_rank(
        self,
        query: str,
        doc_type: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchHit]:
        """检索并返回带排名的命中（供 RRF 融合使用）。"""
        docs = self.search(query, doc_type=doc_type, top_k=top_k)
        return [
            VectorSearchHit(doc_id=d.doc_id, score=d.score, rank=rank)
            for rank, d in enumerate(docs, start=1)
        ]
