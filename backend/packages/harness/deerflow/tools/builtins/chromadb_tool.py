import json
import os
import uuid
from typing import Any, cast

from langchain.tools import tool

from deerflow.chromadb import ChromaDBManager, ChromaConnectionConfig

# ---------- 类型别名 ----------
MetadataValue = str | int | float | bool | None
Metadata = dict[str, MetadataValue]


# ---------- 辅助函数 ----------


def _get_manager() -> ChromaDBManager:
    """从当前进程的环境变量创建 ChromaDBManager 实例。"""
    return ChromaDBManager.from_env(os.environ)


def _connection_info(config: ChromaConnectionConfig) -> dict[str, Any]:
    """将连接配置提取为可序列化的字典，用于工具返回结果中展示连接信息。"""
    return {
        "host": config.host,
        "port": config.port,
        "ssl": config.ssl,
        "tenant": config.tenant,
        "database": config.database,
    }


# ============================================================
# 工具一：chromadb_upsert —— 向 ChromaDB 写入/更新文档
# ============================================================


@tool("chromadb_upsert", parse_docstring=True)
def chromadb_upsert_tool(
    documents: list[str],
    collection_name: str = "default",
    ids: list[str] | None = None,
    metadatas: list[Metadata] | None = None,
) -> str:
    """Store or update documents in a ChromaDB vector database collection for later semantic retrieval.

    Use this tool to persist text content (notes, extracted information, research findings,
    conversation summaries, or any textual data) into ChromaDB so it can be searched
    semantically later using the chromadb_query tool.

    When to use this tool:
    - Saving research findings, extracted data, or notes for future reference
    - Building a knowledge base from documents, web pages, or conversation history
    - Storing structured text with metadata for filtered retrieval
    - Updating previously stored documents with new content (upsert semantics)

    When NOT to use this tool:
    - For temporary data that does not need persistence or semantic search
    - When the user just wants to read or analyze text without storing it
    - For binary files or non-text content (images, PDFs, etc.)

    Notes:
    - Documents are stored using upsert semantics: if an ID already exists, the document is replaced.
    - If no IDs are provided, UUIDs are generated automatically for each document.
    - Attach metadata to documents for fine-grained filtering during queries.
    - The collection is created automatically if it does not exist.

    Args:
        documents: The document texts to store. Each string becomes one searchable document.
        collection_name: Target ChromaDB collection name. Defaults to "default". Use descriptive names to organize different knowledge domains.
        ids: Optional document IDs. If omitted, UUIDs are generated automatically. Provide explicit IDs when you need to update documents later.
        metadatas: Optional metadata list aligned by index with documents. Each dict can contain string, int, float, or bool values for filtering during queries.
    """
    # 参数校验：文档列表不能为空
    if not documents:
        return "Error: `documents` cannot be empty."

    # 参数校验：如果提供了 ids，数量必须与 documents 一致
    if ids is not None and len(ids) != len(documents):
        return "Error: `ids` length must match `documents` length."

    # 参数校验：如果提供了 metadatas，数量必须与 documents 一致
    if metadatas is not None and len(metadatas) != len(documents):
        return "Error: `metadatas` length must match `documents` length."

    # 未提供 ids 时自动生成 UUID
    generated_ids = ids or [str(uuid.uuid4()) for _ in documents]

    # 执行 upsert 操作（插入或更新）
    try:
        manager = _get_manager()
        manager.upsert_texts(
            collection_name=collection_name,
            ids=generated_ids,
            documents=documents,
            metadatas=cast(Any, metadatas),
        )
    except Exception as exc:
        return f"Error upserting into ChromaDB: {exc}"

    # 返回操作结果，包含连接信息和写入详情
    return json.dumps(
        {
            "status": "ok",
            "collection": collection_name,
            "connection": _connection_info(manager.config),
            "upserted_count": len(documents),
            "ids": generated_ids,
        },
        ensure_ascii=False,
    )


# ============================================================
# 工具二：chromadb_query —— 从 ChromaDB 语义检索文档
# ============================================================


@tool("chromadb_query", parse_docstring=True)
def chromadb_query_tool(
    query: str,
    collection_name: str = "default",
    n_results: int = 5,
    where: dict[str, Any] | None = None,
) -> str:
    """Perform semantic search over documents stored in a ChromaDB vector database collection.

    Use this tool to find documents that are semantically similar to a natural language query.
    Results are ranked by relevance (distance), with the most similar documents returned first.

    When to use this tool:
    - Retrieving previously stored knowledge, notes, or research findings
    - Searching for relevant context before answering questions
    - Finding related documents across a knowledge base
    - Looking up information saved in earlier conversations

    When NOT to use this tool:
    - When the information is already available in the current conversation context
    - For exact keyword matching (this performs semantic/meaning-based search)
    - When no documents have been stored yet in the target collection

    Notes:
    - The query is matched semantically, not by exact keywords. Use natural language.
    - Results include document text, metadata, and a distance score (lower = more similar).
    - Use the `where` filter to narrow results by metadata fields (e.g., `{"source": "web"}`).
    - The collection must already exist; querying a non-existent collection will return an error.

    Args:
        query: The natural language query text for semantic retrieval. Describe what you are looking for in plain language.
        collection_name: Target ChromaDB collection name. Defaults to "default". Must match the collection used during storage.
        n_results: Maximum number of documents to retrieve. Defaults to 5. Increase for broader search, decrease for precision.
        where: Optional metadata filter dict for ChromaDB query. Example: `{"category": "research"}` returns only documents with that metadata.
    """
    # 参数校验：查询文本不能为空
    if not query.strip():
        return "Error: `query` cannot be empty."
    # 参数校验：返回数量必须大于 0
    if n_results <= 0:
        return "Error: `n_results` must be greater than 0."

    # 执行语义查询
    try:
        manager = _get_manager()
        raw_results = manager.query_texts(
            collection_name=collection_name,
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        return f"Error querying ChromaDB: {exc}"

    # 解析查询结果：ChromaDB 返回的是嵌套列表结构，取 [0] 获取第一条查询的结果
    result_items: list[dict[str, Any]] = []
    ids = (raw_results.get("ids") or [[]])[0]
    docs = (raw_results.get("documents") or [[]])[0]
    metas = (raw_results.get("metadatas") or [[]])[0]
    dists = (raw_results.get("distances") or [[]])[0]

    # 将各字段按索引组装成结构化的结果列表
    for idx, doc_id in enumerate(ids):
        result_items.append(
            {
                "id": doc_id,
                "document": docs[idx] if idx < len(docs) else None,
                "metadata": metas[idx] if idx < len(metas) else None,
                "distance": dists[idx] if idx < len(dists) else None,
            }
        )

    # 返回查询结果，包含连接信息、查询参数和匹配文档
    return json.dumps(
        {
            "status": "ok",
            "collection": collection_name,
            "connection": _connection_info(manager.config),
            "query": query,
            "count": len(result_items),
            "results": result_items,
        },
        ensure_ascii=False,
    )
